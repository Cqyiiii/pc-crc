"""Replay Table 4, the pooled entropy policies in Table 1, and the Native comparison."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tail_crc import crc_threshold


SEED = 20261122
RECEIPT_FEATURES = ("box_sum_nll", "ans_mean_nll", "ans_sum_nll", "ans_len", "tail10_nll")
PANEL_SIZES = {"DAPO": 500, "Omni": 300}


def risks(rows, feature):
    """A higher value denotes greater risk. An invalid answer can never pass a strict cut."""
    values = np.array([r[feature] if r["vs"] != "U" else np.inf for r in rows], dtype=float)
    valid = np.array([r["vs"] != "U" for r in rows])
    assert np.isfinite(values[valid]).all()
    if feature == "tail_margin":
        values[valid] *= -1
    return values


def weighted_auc(scores, labels, weights):
    """Wrong-positive weighted Mann–Whitney AUROC on valid submissions, with half credit for ties."""
    valid = np.flatnonzero(labels != "U")
    order = valid[np.argsort(scores[valid], kind="stable")]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(scores[order]))]
    wrong = labels[order] == "W"
    obs = weights[:, order]
    w = np.add.reduceat(obs * wrong, starts, axis=1)
    r = np.add.reduceat(obs * ~wrong, starts, axis=1)
    numerator = (w * (np.cumsum(r, axis=1) - r / 2)).sum(axis=1)
    denominator = w.sum(axis=1) * r.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(obs), np.nan), where=denominator > 0)


def crc_pass(cal, test_scores, name, alpha):
    """Split CRC over all CAL families, with tied values at the cutoff rejected together."""
    threshold = crc_threshold(zip((r["vs"] for r in cal), risks(cal, name)), alpha)
    return test_scores < threshold


def entropy_policies(groups, out_path):
    points = []
    for domain in ("Omni", "DAPO"):
        for size in ("4B", "8B"):
            for budget in (4096, 8192):
                for alpha in (0.065, 0.1):
                    seeds = []
                    for arm in ("G0_41", "G0_73"):
                        cell = groups[(size, arm, domain, budget)]
                        test = sorted(cell["TEST"], key=lambda r: r["family"])
                        families = [r["family"] for r in test]
                        if seeds:
                            assert families == previous
                        previous = families
                        accept = crc_pass(cell["CAL"], risks(test, "tail_H"), "tail_H", alpha)
                        vs = np.array([r["vs"] for r in test])
                        strict = np.array([r["strict"] == "correct" for r in test])
                        seeds.append(np.column_stack((
                            accept & (vs == "R"), accept & (vs == "W"), accept & strict,
                            [r["natural_close"] for r in test], [r["u_reason"] == "trunc" for r in test],
                            [r["output_tokens"] for r in test], [r["usd"] for r in test],
                        )))
                    pooled = np.mean(seeds, axis=0)
                    right, wrong, strict, natural, trunc, tokens, usd = pooled.mean(axis=0)
                    coverage = right + wrong
                    rates = dict(R_pct=right, W_pct=wrong, U_pct=1-coverage, Net_pct=right-wrong,
                                 Cov_pct=coverage, Prec_pct=right/coverage, strict_R_pct=strict,
                                 nat_close_pct=natural, trunc_pct=trunc)
                    points.append(dict(domain=domain, size=size, decode="ExtBF", arm="G0_pooled",
                                       budget=budget, filter_rule=f"crc_tailH_{alpha}",
                                       **{k: round(100*v, 6) for k, v in rates.items()},
                                       avg_out_tok=tokens, usd_per_prob=usd,
                                       source_study="A2 frozen per-seed cuts + released TEST family join"))
                    if (domain, size, budget, alpha) == ("DAPO", "8B", 8192, 0.065):
                        headline = (families, pooled[:, :2])
    write_table(out_path / "whitebox_points.csv", points)
    return headline


def native_comparison(rows, families, pooled):
    native = sorted((r for r in rows if (r["domain"], r["phase"], r["size"], r["decode"],
                                         r["arm"], r["budget"]) == ("DAPO", "TEST", "8B", "Native", "base", 8192)),
                    key=lambda r: r["family"])
    assert [r["family"] for r in native] == families
    labels = np.array([r["vs"] for r in native])
    delta = pooled - np.column_stack((labels == "R", labels == "W"))
    delta = np.column_stack((delta, delta[:, 0] - delta[:, 1]))
    n = len(families)
    rng = np.random.default_rng(SEED)
    # Generate all 20,000 paired draws in sequence, bounding temporary memory.
    means = np.concatenate([rng.multinomial(n, np.full(n, 1/n), size=1000) @ delta / n
                            for _ in range(20)])
    lower, upper = np.quantile(means, [0.025, 0.975], axis=0)
    metrics = {k: dict(point=p, lo=l, hi=h)
               for k, p, l, h in zip(("R", "W", "net"), delta.mean(axis=0), lower, upper)}
    return dict(scope="post hoc; DAPO TEST500; pooled 8B ExtBF 8192 tail_H alpha=.065 minus unfiltered "
                      "Native base 8192; fixed cuts, family-paired percentile bootstrap; rates as fractions",
                bootstrap=20000, seed=SEED, metrics=metrics)


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(rows, out_path):
    """Write the three outputs from the released scalar observations in memory."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("wb_rank") is not None:
            assert r["decode"] == "BF" and r["domain"] in PANEL_SIZES
            groups[(r["size"], r["arm"], r["domain"], r["budget"])][r["phase"]].append(r)
    assert len(groups) == 36
    table = []
    weights = {}
    for (size, arm, domain, budget), cell in sorted(groups.items()):
        n = PANEL_SIZES[domain]
        for phase in ("CAL", "TEST"):
            cell[phase].sort(key=lambda r: r["wb_rank"])
            assert [r["wb_rank"] for r in cell[phase]] == list(range(n))
            assert len({r["family"] for r in cell[phase]}) == n
        cal, test = cell["CAL"], cell["TEST"]
        assert {r["family"] for r in cal}.isdisjoint(r["family"] for r in test)
        if n not in weights:
            draws = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=2000)
            weights[n] = np.vstack((np.ones(n, dtype=int), draws))
        labels = np.array([r["vs"] for r in test])
        w = weights[n]
        rec = dict(size=size, arm=arm, domain=domain, budget=budget, n_test=n,
                   n_valid=int((labels != "U").sum()), n_wrong=int((labels == "W").sum()))
        for name in RECEIPT_FEATURES:
            rec[name + "_auc"] = weighted_auc(risks(test, name), labels, w[:1])[0]
        # receipt_tail is the value stored by the white-box trace extractor. Use it
        # alongside entropy for the paired difference and the white-box policy replay.
        h, receipt = (risks(test, name) for name in ("tail_H", "receipt_tail"))
        h_auc = weighted_auc(h, labels, w)
        diff = h_auc - weighted_auc(receipt, labels, w)
        ci = np.nanquantile(diff[1:], [0.025, 0.975])  # one-class draws have undefined AUROC
        rec.update(tail_H_auc=h_auc[0], tail_margin_auc=weighted_auc(risks(test, "tail_margin"), labels, w[:1])[0],
                   H_minus_receipt=diff[0], diff_lower=ci[0], diff_upper=ci[1])
        for lab, feat, score in (("tailH", "tail_H", h), ("tail10", "receipt_tail", receipt)):
            passed = crc_pass(cal, score, feat, 0.065)
            for outcome in ("R", "W"):
                rec[f"crc65_{lab}_{outcome}"] = int((passed & (labels == outcome)).sum())
        table.append(rec)
    write_table(out_path / "table4_uncertainty_auroc_and_whitebox.csv", table)
    families, pooled = entropy_policies(groups, out_path)
    pair = native_comparison(rows, families, pooled)
    (out_path / "whitebox_pair.json").write_text(json.dumps(pair, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
