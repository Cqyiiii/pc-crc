"""Replay the released DAPO and Omni operating points and fixed-cut paired intervals.

Rows have one record per family, policy, and budget. Calibration is per seed.
A pooled row averages the two seed outcomes *within* a family before resampling.
"""
from collections import defaultdict
import csv
from functools import lru_cache
import json
import math

import numpy as np


BUDGETS = (4096, 8192)
ALPHAS = (.04, .065, .08, .10, .125, .15)
PRIMARY = (.065, .10)
SEEDS = ("G0_41", "G0_73")
SCORES = ("tail10_nll", "box_sum_nll", "ans_mean_nll", "ans_sum_nll", "ans_len")
ARMS = {
    "4B": ("base", "G0_41", "G0_73", "E1_41"),
    "8B": ("base", "G0_41", "G0_73", "G1_41", "M5_41"),
}
# Ordering in the Omni registration and released CSV, including reference arms.
OMNI_ARMS = {
    "8B": ("G0_41", "G1_41", "M5_41", "base", "G0_73"),
    "4B": ("base", "E1_41", "G0_41", "G0_73"),
}
NAMES = ("R", "W", "U", "net", "coverage", "strict_R", "strict_W", "natural",
         "trunc", "abstain", "eos_no_box", "output", "usd")
INDEX = {name: i for i, name in enumerate(NAMES)}
SUP = "paper_final_supplement_v1"
OOD = "pc_crc_ood_confirm_v1"
TWO_A = "pc_crc_confirm_2a_v1"
PAIRED = {
    SUP: ("R", "W", "net", "usd", "coverage", "trunc", "abstain"),
    OOD: ("R", "W", "net", "coverage", "usd", "strict_R", "tokens"),
    TWO_A: ("R", "W", "net", "coverage", "strict_R", "usd"),
}


def key(*parts):
    return "/".join(map(str, parts))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Panel:
    """One arm on one split, ordered by the shared anonymous family ID."""
    def __init__(self, records):
        self.labels = np.array([x["vs"] for x in records])
        self.records = records
        self.family = tuple(x["family"] for x in records)

    @lru_cache(maxsize=None)
    def score(self, name):
        # An unavailable score on an uncommitted output never passes strict < cut.
        valid = self.labels != "U"
        result = np.full(len(valid), np.inf)
        for i in np.flatnonzero(valid):
            value = self.records[i][name]
            if value is None or not np.isfinite(value):
                raise ValueError(f"Missing or nonfinite {name} for a committed answer")
            result[i] = value
        return result

    @lru_cache(maxsize=None)
    def wrong_scores(self, score):
        return np.sort(self.score(score)[self.labels == "W"])

    def cut(self, score, alpha):
        allowed = math.floor((len(self.labels) + 1) * alpha - 1 + 1e-10)
        assert allowed >= 0
        wrong = self.wrong_scores(score)
        return wrong[allowed] if allowed < len(wrong) else np.inf

    @lru_cache(maxsize=None)
    def raw(self):
        r, w = self.labels == "R", self.labels == "W"
        valid = r | w
        strict = np.array([x["strict"] for x in self.records])
        reason = np.array([x["u_reason"] for x in self.records], dtype=object)
        return np.column_stack((
            r, w, ~valid, r.astype(float) - w, valid, valid & (strict == "correct"),
            valid & (strict == "wrong"), [x["natural_close"] for x in self.records],
            reason == "trunc", reason == "abstain", reason == "eos_no_box",
            [x["output_tokens"] for x in self.records], [x["usd"] for x in self.records],
        )).astype(float)

    def filtered(self, score, cut):
        values = self.raw().copy()
        reject = self.score(score) >= cut
        # Each rejected submission becomes U. Generation outcomes and fees are fixed.
        values[reject, :7] = (0, 0, 1, 0, 0, 0, 0)
        return values


def operating_point(matrix):
    p = dict(zip(NAMES, matrix.mean(axis=0).tolist()))
    p["precision"] = p["R"] / p["coverage"] if p["coverage"] else None
    p["conditional_error"] = p["W"] / p["coverage"] if p["coverage"] else None
    return p


def t1_row(domain, size, decode, arm, budget, rule, p, study):
    metrics = {
        "R_pct": "R", "W_pct": "W", "U_pct": "U", "Net_pct": "net",
        "Cov_pct": "coverage", "Prec_pct": "precision", "strict_R_pct": "strict_R",
        "nat_close_pct": "natural", "trunc_pct": "trunc",
    }
    return dict(domain=domain, size=size, decode="ExtBF" if decode == "BF" else decode,
                arm=arm, budget=budget, filter_rule=rule,
                **{k: round(100 * p[v], 6) if p[v] is not None else None
                   for k, v in metrics.items()}, avg_out_tok=p["output"],
                usd_per_prob=p["usd"], source_study=study)


def paired_rows(study, panel, pairs, seed):
    """Percentile intervals from 20,000 family resamples, conditional on CAL fits."""
    metrics = PAIRED[study]
    columns = [INDEX["output" if m == "tokens" else m] for m in metrics]
    n = len(pairs[0][1])
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), 20000).astype(float)
    result = []
    # Share weights across contrasts and batch the products to bound working memory.
    for start in range(0, len(pairs), 10):
        batch = pairs[start:start + 10]
        differences = np.column_stack([(left - right)[:, columns] for _, left, right in batch])
        point = differences.mean(axis=0)
        lo, hi = np.quantile((weights @ differences) / n, [.025, .975], axis=0)
        for j, (contrast, _, _) in enumerate(batch):
            for i, metric in enumerate(metrics, start=j * len(metrics)):
                result.append(dict(study=study, panel=panel, contrast=contrast, metric=metric,
                                   estimate=float(point[i]), ci95_lo=float(lo[i]), ci95_hi=float(hi[i])))
    return result


def supplement(panels, domain):
    t1, t3, pairs, frontier = [], [], [], []
    for size, arms in ARMS.items():
        table = {}
        for budget in BUDGETS:
            def cell(arm, alpha=0, decode="BF"):
                test = panels[domain, "TEST", size, arm, budget, decode]
                if not alpha:
                    return test.raw()
                cal = panels[domain, "CAL", size, arm, budget, decode]
                return test.filtered("tail10_nll", cal.cut("tail10_nll", alpha))

            for arm in ("base", "G0_41"):
                table[key("Native", arm, budget)] = cell(arm, decode="Native")
            for arm in arms:
                table[key("BF", arm, budget)] = cell(arm)
            for alpha in (0, *ALPHAS):
                suffix = f"/crc{alpha}" if alpha else ""
                table[key("BF", "G0_pooled", budget) + suffix] = (cell(SEEDS[0], alpha) + cell(SEEDS[1], alpha)) / 2
            if size == "8B":
                if budget == 4096:
                    for arm in ("G0_41", "G1_41"):
                        for alpha in PRIMARY:
                            table[key("BF", arm, budget, f"crc{alpha}")] = cell(arm, alpha)
                for arm in ("S3A", "S3B"):
                    table[key("BF", arm, budget)] = cell(arm)
                    if budget == 4096:
                        for alpha in PRIMARY:
                            table[key("BF", arm, budget, f"crc{alpha}")] = cell(arm, alpha)
                # Vary the allowed calibration-wrong count through the observed range.
                solos = [(panels[domain, "CAL", size, a, budget, "BF"].wrong_scores("tail10_nll"),
                          panels[domain, "TEST", size, a, budget, "BF"]) for a in SEEDS]
                n = len(solos[0][1].family)
                for k in range(1 + max(len(wrong) for wrong, _ in solos)):
                    accepted = wrong_count = 0
                    for wrong, test in solos:
                        cut = wrong[k] if k < len(wrong) else np.inf
                        on = test.score("tail10_nll") < cut
                        accepted += int(on.sum())
                        wrong_count += int((on & (test.labels == "W")).sum())
                    frontier.append(dict(domain=domain, budget=budget, k=k, alpha=(1+k)/(n+1),
                                         coverage=accepted/(2*n), W=wrong_count/(2*n),
                                         conditional_error=wrong_count/accepted if accepted else None))

        def compare(left, right):
            pairs.append((key(domain, size, left) + " - " + right, table[left], table[right]))

        for budget in BUDGETS:
            compare(key("Native", "G0_41", budget), key("Native", "base", budget))
            for arm in ("base", "G0_41"):
                compare(key("BF", arm, budget), key("Native", arm, budget))
            for suffix in ("", "/crc0.065", "/crc0.1"):
                left = key("BF", "G0_pooled", budget) + suffix
                compare(left, key("Native", "base", budget))
                if budget == 4096:
                    compare(left, key("Native", "base", 8192))
            if size == "8B":
                for arm in ("S3A", "S3B"):
                    for ref in ("G0_41", "G1_41"):
                        compare(key("BF", arm, budget), key("BF", ref, budget))
                    if budget == 4096:
                        for alpha in PRIMARY:
                            for ref in ("G0_pooled", "G0_41", "G1_41"):
                                compare(key("BF", arm, budget, f"crc{alpha}"), key("BF", ref, budget, f"crc{alpha}"))
                compare(key("BF", "S3A", budget), key("BF", "S3B", budget))

        for label, matrix in table.items():
            decode, arm, budget, *rule = label.split("/")
            p = operating_point(matrix)
            t1.append(t1_row(domain, size, decode, arm, budget,
                             rule[0].replace("crc", "crc_tail10_") if rule else "raw", p,
                             SUP + "/results/final.json"))
            if (size == "8B" and decode == "BF" and arm in ("S3A", "S3B", "G0_pooled", "G0_41", "G1_41")
                    and (not rule or rule[0] in ("crc0.065", "crc0.1"))):
                t3.append(dict(cell=key(domain, size, label), **p))
    t2 = paired_rows(SUP, "in_cell_key", pairs, 20261122)
    net = {r["contrast"]: r for r in t2 if r["metric"] == "net"}
    for row in t3:
        _, _, _, _, budget, *rule = row["cell"].split("/")
        contrast = row["cell"] + " - " + key("BF", "G0_pooled", budget, *rule)
        delta = net.get(contrast, {})
        row.update(delta_net_vs_pooled_same_CRC=delta.get("estimate"), delta_lo=delta.get("ci95_lo"),
                   delta_hi=delta.get("ci95_hi"))
    return t1, t2, t3, frontier


def track_rules():
    yield "raw", "box_sum_nll", 0
    for track, alphas in (("A", PRIMARY), ("B", ALPHAS)):
        for score in SCORES:
            for alpha in alphas:
                yield track, score, alpha


def omni_tracks(panels):
    table, rows, pairs = {}, [], []

    def store(size, budget, arm, track, score, alpha, matrix):
        table[key(size, budget, arm, track, score, alpha)] = matrix
        if track != "raw":
            n = len(matrix)
            r, w = matrix[:, :2].sum(0) / n
            rows.append(dict(size=size, budget=budget, arm=arm, track=track, score=score,
                             alpha=alpha, n=n, R=r, W=w, net=r-w, coverage=r+w,
                             precision=r/(r+w) if r+w else None))

    for size, arms in OMNI_ARMS.items():
        for arm in arms:
            for budget in BUDGETS:
                test = panels["Omni", "TEST", size, arm, budget, "BF"]
                for track, score, alpha in track_rules():
                    values = test.raw()
                    if track != "raw":
                        source = "DAPO" if track == "A" else "Omni"
                        cal = panels[source, "CAL", size, arm, budget, "BF"]
                        values = test.filtered(score, cal.cut(score, alpha))
                    store(size, budget, arm, track, score, alpha, values)

    for size, arms in OMNI_ARMS.items():
        for budget in BUDGETS:
            for track, score, alpha in track_rules():
                tag = key(track, score, alpha)
                raw_tag = key("raw", "box_sum_nll", 0)
                x, y = (table[key(size, budget, a, tag)] for a in SEEDS)
                store(size, budget, "G0_pooled", track, score, alpha, (x+y)/2)
                if track != "raw" and (track != "B" or score != "tail10_nll" or alpha not in PRIMARY):
                    continue
                left = key(size, budget, "G0_pooled", tag)
                refs = []
                if track == "B":
                    refs.append(key(size, budget, "G0_pooled", raw_tag))
                for arm in arms:
                    if arm not in SEEDS:
                        refs.append(key(size, budget, arm, raw_tag))
                        if track == "B":
                            refs.append(key(size, budget, arm, tag))
                for right in refs:
                    pairs.append((left + " - " + right, table[left], table[right]))
    return paired_rows(OOD, "Omni_TEST300", pairs, 20261122), rows


def original_2a(panels, domain):
    """Both evaluated panels use cuts from the original 500-family DAPO CAL scores."""
    rows, pairs = [], []
    for size in ("8B", "4B") if domain == "DAPO" else ("8B",):
        refs = ("base", "M5_41", "G1_41") if size == "8B" else ("base", "E1_41")
        for budget in BUDGETS:
            table = {}
            for arm in ARMS[size]:
                test = panels[domain, "TEST", size, arm, budget, "BF"]
                cal = panels["DAPO", "CAL", size, arm, budget, "BF"]
                table[arm, "raw"] = test.raw()
                for alpha in ALPHAS:
                    score = "box_sum_nll_2a"
                    table[arm, f"crc{alpha}"] = test.filtered(score, cal.cut(score, alpha))
            for rule in ("raw", *(f"crc{alpha}" for alpha in ALPHAS)):
                pooled = (table[SEEDS[0], rule] + table[SEEDS[1], rule]) / 2
                left = key(size, "G0_pooled", budget, rule)
                for ref in refs:
                    for ref_rule in dict.fromkeys(("raw", rule)):
                        right = key(size, ref, budget, ref_rule)
                        pairs.append((left + " - " + right, pooled, table[ref, ref_rule]))
                if domain == "DAPO" and rule != "raw":
                    rows.append(t1_row(domain, size, "BF", "G0_pooled", budget, rule.replace("crc", "crc_box_"),
                                       operating_point(pooled), TWO_A + "/reports/results.json"))
    return rows, paired_rows(TWO_A, "TEST" if domain == "DAPO" else "OOD", pairs, 20261121)


def run(rows, out):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["domain"], r["phase"], r["size"], r["arm"], r["budget"], r["decode"]].append(r)
    panels = {}
    families = {}
    for cell, records in grouped.items():
        panel = Panel(sorted(records, key=lambda r: r["family"]))
        domain, phase = cell[:2]
        assert len(panel.family) == len(set(panel.family)) == {"DAPO": 500, "Omni": 300, "OmniEnriched": 200}[domain]
        assert families.setdefault((domain, phase), panel.family) == panel.family
        panels[cell] = panel
    for domain in ("DAPO", "Omni"):
        assert set(families[domain, "CAL"]).isdisjoint(families[domain, "TEST"])

    t1, t2, t3, frontier = [], [], [], []
    for domain in ("DAPO", "Omni"):
        points, comparisons, ablation, trace = supplement(panels, domain)
        t1.extend(points)
        t2.extend(comparisons)
        t3.extend(ablation)
        frontier.extend(trace)
    contrasts, t5 = omni_tracks(panels)
    t2.extend(contrasts)
    for domain in ("DAPO", "OmniEnriched"):
        points, contrasts = original_2a(panels, domain)
        t1.extend(points)
        t2.extend(contrasts)
    assert tuple(map(len, (t1, t2, t3, t5, frontier))) == (156, 2382, 44, 880, 455)
    write_csv(out / "main_points.csv", t1)
    write_csv(out / "table2_paired_contrasts_and_cost.csv", t2)
    write_csv(out / "table3_penalty_ablation_8b.csv", t3)
    write_csv(out / "table5_trackA_vs_trackB_and_ltt.csv", t5)
    (out / "frontier.json").write_text(json.dumps(frontier, indent=2, allow_nan=False) + "\n")
    # Release retained panels if run() is called again in the same interpreter.
    Panel.score.cache_clear()
    Panel.wrong_scores.cache_clear()
    Panel.raw.cache_clear()
