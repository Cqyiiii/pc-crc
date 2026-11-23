"""Build the twelve manuscript tables after reproduce.py, then check every displayed row."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
INPUTS = PAPER / "table_inputs"
REFERENCES = PAPER / "references" / "tables"
DOMAINS = ("DAPO", "Omni")
MODELS = ("4B", "8B")
CAPS = {"4096": "4K", "8192": "8K"}
ARM_LABELS = {"G0_pooled": "Binary RL", "G1_41": "Penalty RL",
              "S3A": "Explicit-Abstain RL", "S3B": "Mild-Penalty RL"}
A3_ARMS = {"G0_41": "G0_41", "G0_73": "G0_73", "G1_41": "G1_41",
           "G0_KL_41": "G0+KL", "G0_multi_41": "G0+multi"}
T1 = "table1_dapo_omni_main_results.csv"
T2 = "table2_paired_contrasts_and_cost.csv"
T3 = "table3_penalty_ablation_8b.csv"
T4 = "table4_uncertainty_auroc_and_whitebox.csv"
T5 = "table5_trackA_vs_trackB_and_ltt.csv"
T6 = "table6_a100_portability_latency_and_pilot.csv"

COLUMNS = {
    1: "domain model cap native_R_pct native_W_pct budget_forcing_R_pct budget_forcing_W_pct "
       "answer_sft_R_pct answer_sft_W_pct binary_rl_R_pct binary_rl_W_pct tail_crc_R_pct tail_crc_W_pct "
       "binary_rl_cost_saving_pct",
    2: "domain arm raw_R_pct raw_W_pct truncation_pct explicit_abstention_pct net_alpha_065_pct net_alpha_10_pct",
    3: "model generation_configuration seconds_per_family comparison_with_native_8K",
    4: "approach decision_and_risk_target deployment_information citation_url",
    5: "domain model cap delta_R_vs_base_pp base_ci95_lo_pp base_ci95_hi_pp delta_R_vs_sft_pp "
       "sft_ci95_lo_pp sft_ci95_hi_pp binary_rl_usd_per_family",
    6: "domain model cap native_base_R_pct native_binary_rl_seed_41_R_pct delta_R_pp ci95_lo_pp ci95_hi_pp",
    7: "domain model arm cap box_sum_auc answer_sum_auc answer_mean_auc tail_nll_auc tail_entropy_auc "
       "entropy_minus_nll_auc ci95_lo ci95_hi",
    8: "domain score cal_issuance test_empirical_meets_target test_upper95_meets_target",
    9: "domain model cap alpha cal_n_min cal_n_max mean_W_min_cal_pct mean_W_max_cal_pct sd_min_cal_pp "
       "sd_max_cal_pp sd_nonincreasing",
    10: "model local_arm updates domain cap raw_R raw_W tail_crc_R tail_crc_W delta_raw_R_pp ci95_lo_pp ci95_hi_pp",
    12: "model domain maj2_R maj2_W agreement_R agreement_W single_8K_R single_8K_W two_4K_tokens one_8K_tokens",
}
COLUMNS[11] = COLUMNS[10]


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def keyed(rows, *fields):
    index = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError(f"Duplicate record for {fields}: {key}")
        index[key] = row
    return index


def pp_interval(row):
    return [f"{100 * float(row['estimate']):+.2f}", f"{100 * float(row['ci95_lo']):.2f}",
            f"{100 * float(row['ci95_hi']):.2f}"]


def operating_points(out):
    points = keyed(read(out / T1), "domain", "size", "decode", "arm", "budget", "filter_rule")
    contrasts = keyed(read(out / T2), "study", "panel", "contrast", "metric")
    tracks = keyed(read(out / T5), "size", "budget", "arm", "track", "score", "alpha")
    main, raw_contrasts, native_transfer = [], [], []
    for domain, size, budget in product(DOMAINS, MODELS, CAPS):
        def point(arm, decode="ExtBF", rule="raw"):
            return points[domain, size, decode, arm, budget, rule]

        sft = "E1_41" if size == "4B" else "M5_41"
        native, base = point("base", "Native"), point("base")
        tuned, binary = point(sft), point("G0_pooled")
        tail = point("G0_pooled", rule="crc_tail10_0.065")
        if domain == "Omni":
            target = tracks[size, budget, "G0_pooled", "B", "tail10_nll", "0.065"]
            for metric in ("R", "W"):
                # T1 stores percentages to six decimal places.
                assert abs(float(tail[metric + "_pct"]) - 100 * float(target[metric])) <= 5.1e-7
        key = [domain, size, CAPS[budget]]
        rates = [f"{float(row[column]):.1f}" for row in (native, base, tuned, binary, tail)
                 for column in ("R_pct", "W_pct")]
        cost = float(binary["usd_per_prob"])
        savings = 100 * (1 - cost / float(base["usd_per_prob"]))
        main.append([*key, *rates, f"{savings:.1f}"])
        intervals = []
        for control in ("base", sft):
            if domain == "DAPO":
                contrast = f"{size}/G0_pooled/{budget}/raw - {size}/{control}/{budget}/raw"
                lookup = ("pc_crc_confirm_2a_v1", "TEST", contrast, "R")
            else:
                prefix = f"{size}/{budget}/"
                suffix = "/raw/box_sum_nll/0"
                contrast = f"{prefix}G0_pooled{suffix} - {prefix}{control}{suffix}"
                lookup = ("pc_crc_ood_confirm_v1", "Omni_TEST300", contrast, "R")
            intervals.extend(pp_interval(contrasts[lookup]))
        raw_contrasts.append([*key, *intervals, f"{cost:.6f}"])
        native_seed = point("G0_41", "Native")
        contrast = f"{domain}/{size}/Native/G0_41/{budget} - Native/base/{budget}"
        interval = contrasts["paper_final_supplement_v1", "in_cell_key", contrast, "R"]
        native_transfer.append([*key, f"{float(native['R_pct']):.2f}",
                                f"{float(native_seed['R_pct']):.2f}", *pp_interval(interval)])
    return main, raw_contrasts, native_transfer


def penalties(out):
    cells = keyed(read(out / T3), "cell")
    rows = []
    for domain, arm in product(DOMAINS, ARM_LABELS):
        key = f"{domain}/8B/BF/{arm}/4096"
        raw = cells[(key,)]
        values = [raw[k] for k in ("R", "W", "trunc", "abstain")]
        values.extend(cells[(f"{key}/crc{alpha}",)]["net"] for alpha in ("0.065", "0.1"))
        rows.append([domain, ARM_LABELS[arm], *[f"{100 * float(v):.1f}" for v in values]])
    return rows


def generation_time(out):
    wall = [r for r in read(out / T6) if (r["category"], r["engine"]) == ("WallClock", "vLLM_A100")]
    by_run = keyed(wall, "model", "arm", "decode_or_test")
    rows = []
    for size in MODELS:
        def seconds(arm, mode):
            row = by_run[size, arm, f"{mode}_DAPO"]
            name, value = row["metric_2"].split("=")
            assert name == "batched_wall_s_per_problem" and int(row["n"]) == 50
            return float(value)

        baseline = seconds("base", "native8192")
        short = seconds("G0_41", "ext4096")
        rows.append([size, "Native, 8K", f"{baseline:.2f}", ""])
        rows.append([size, "Binary RL, 4K", f"{short:.2f}", f"{baseline / short:.2f}× speedup"])
        if size == "8B":
            long = seconds("G0_41", "ext8192")
            rows.append([size, "Binary RL, 8K", f"{long:.2f}", f"{100 * (1 - long / baseline):.2f}% less time"])
    return rows


def methods():
    rows = []
    for r in read(INPUTS / "method_comparison.csv"):
        cite = f" [{r['paper_reference']}]" if r["paper_reference"] else ""
        rows.append([r["approach"] + cite, r["decision_and_risk_target"], r["deployment_information"], r["source_url"]])
    return rows


def uncertainty(out):
    features = ("box_sum_nll_auc", "ans_sum_nll_auc", "ans_mean_nll_auc", "tail10_nll_auc", "tail_H_auc")
    cells = keyed(read(out / T4), "size", "arm", "domain", "budget")
    rows = []
    for (size, arm, domain, budget), r in sorted(cells.items()):
        assert 0 < int(r["n_wrong"]) < int(r["n_valid"]) <= int(r["n_test"])
        estimates = [f"{float(r[feature]):.3f}" for feature in features]
        rows.append([domain, size, arm, CAPS[budget], *estimates, f"{float(r['H_minus_receipt']):+.3f}",
                     f"{float(r['diff_lower']):.3f}", f"{float(r['diff_upper']):.3f}"])
    return rows


def certificates(out):
    candidates = read(out / "ltt_all.csv")
    keyed(candidates, "domain", "size", "arm", "budget", "score", "target")
    groups = defaultdict(list)
    for r in candidates:
        groups[r["domain"], r["score"]].append(r)
    scores = {"tail10_nll": "Tail NLL", "ans_mean_nll": "Answer-mean NLL"}
    rows = []
    for domain, score in product(DOMAINS, scores):
        group = groups.pop((domain, score))
        assert len(group) == 72 and all(r["issued"] in {"True", "False"} for r in group)
        issued = [r for r in group if r["issued"] == "True"]
        for r in group:
            if r["issued"] == "False":
                assert r["test_accepted"] == r["test_error"] == r["test_upper95"] == ""
            elif int(r["test_accepted"]) == 0:
                assert r["test_error"] == r["test_upper95"] == ""
        evaluable = [r for r in issued if int(r["test_accepted"]) > 0]
        passed = [sum(float(r[col]) <= float(r["target"]) + 1e-14 for r in evaluable)
                  for col in ("test_error", "test_upper95")]
        rows.append([domain, scores[score], f"{len(issued)}/{len(group)}",
                     *[f"{n}/{len(evaluable)}" for n in passed]])
    assert not groups
    return rows


def calibration_sizes(out):
    curves = defaultdict(list)
    for r in read(out / "ncal_all.csv"):
        curves[r["domain"], r["size"], r["budget"], r["alpha"]].append(r)
    rows = []
    for (domain, size, budget, alpha), curve in sorted(curves.items()):
        ordered = sorted(curve, key=lambda r: int(r["ncal"]))
        ns = [int(r["ncal"]) for r in ordered]
        assert ns == ([50, 100, 200, 300, 500] if domain == "DAPO" else [50, 100, 200, 300])
        sd = [float(r["sd"]) for r in ordered]
        ends = [f"{100 * float(r[column]):.3f}" for column in ("mean", "sd") for r in (ordered[0], ordered[-1])]
        rows.append([domain, size, CAPS[budget], alpha, ns[0], ns[-1], *ends,
                     "Yes" if all(a >= b for a, b in zip(sd, sd[1:])) else "No"])
    return rows


def local_pilots():
    cells = keyed(read(INPUTS / "a3_final_tier.csv"), "decoder", "model", "arm", "domain", "budget")
    rows = {"ExtBF": [], "Native": []}
    for (decoder, model, arm, domain, budget), r in sorted(cells.items()):
        n = int(r["n_test"])
        steps = int(r["updates"])
        assert r["engine"] == "vllm" and int(r["n_cal"]) == n == 16
        assert steps == (25 if model == "Llama" else 12)
        assert int(r["training_rollouts"]) == 16 * steps and int(r["training_family_exposures"]) == 4 * steps
        reference = "base" if arm in ("G0_41", "G0_73") else f"local_G0_41_s{steps:03}"
        assert r["paired_reference"] == reference
        right, wrong, crc_r, crc_w, ref_r = [int(r[k]) for k in ("raw_R", "raw_W", "crc065_R", "crc065_W",
                                                                 "paired_reference_raw_R")]
        assert right + wrong <= n and 0 <= crc_r <= right and 0 <= crc_w <= wrong and 0 <= ref_r <= n
        delta = (right - ref_r) / n
        rows[decoder].append([model, A3_ARMS[arm], steps, domain, CAPS[budget], right, wrong, crc_r, crc_w,
                              f"{100 * delta:+.2f}", f"{100 * float(r['ci95_lo']):.2f}",
                              f"{100 * float(r['ci95_hi']):.2f}"])
    return rows["ExtBF"], rows["Native"]


def paired_draws(data):
    groups = defaultdict(list)
    for r in read(data / "system_vote_families.csv"):
        groups[r["model"], r["domain"]].append(r)
    rows, errors = [], {}
    for model, domain in product(MODELS, DOMAINS):
        group = groups.pop((model, domain))
        assert len(group) == len({r["family"] for r in group}) == 16
        assert all(r["answers_agree"] in ("0", "1") for r in group)
        assert all(r[k] in ("R", "W", "U") for r in group for k in ("draw0_grade", "draw1_grade", "single_grade"))
        # Under the stated two-draw tie break, the first draw always supplies the chosen answer.
        majority = Counter(r["draw0_grade"] for r in group)
        agree = Counter(r["draw0_grade"] for r in group if r["answers_agree"] == "1")
        single = Counter(r["single_grade"] for r in group)
        accepted = agree["R"] + agree["W"]
        errors[f"{model}/{domain}"] = agree["W"] / accepted if accepted else None
        pair_tokens = sum(int(r["draw0_generated"]) + int(r["draw1_generated"]) for r in group)
        single_tokens = sum(int(r["single_generated"]) for r in group)
        counts = [policy[grade] for policy in (majority, agree, single) for grade in ("R", "W")]
        rows.append([model, domain, *counts, pair_tokens, single_tokens])
    assert not groups
    return rows, errors


def build(out, data):
    tables = {}
    tables[1], tables[5], tables[6] = operating_points(out)
    tables[2] = penalties(out)
    tables[3] = generation_time(out)
    tables[4] = methods()
    tables[7] = uncertainty(out)
    tables[8] = certificates(out)
    tables[9] = calibration_sizes(out)
    tables[10], tables[11] = local_pilots()
    tables[12], agreement_errors = paired_draws(data)
    dest = out / "paper" / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    report = {"comparison": "exact strings at manuscript display precision, including labels and row order",
              "reference": "paper/references/tables", "tables": {}, "agreement_conditional_error": agreement_errors}
    mismatches = []
    for num, rows in sorted(tables.items()):
        name = f"table_{num:02}.csv"
        header = COLUMNS[num].split()
        with (dest / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        with (REFERENCES / name).open(newline="", encoding="utf-8") as f, (dest / name).open(newline="", encoding="utf-8") as g:
            expected, actual = list(csv.reader(f)), list(csv.reader(g))
        if len(actual) != len(expected):
            mismatches.append(f"{name}: {len(actual)-1} rows, expected {len(expected)-1}")
        for index, (target, observed) in enumerate(zip(expected, actual)):
            if len(observed) != len(target):
                mismatches.append(f"{name} row {index}: expected {len(target)} fields, found {len(observed)}")
            for col, (a, b) in enumerate(zip(target, observed)):
                if a != b:
                    mismatches.append(f"{name} row {index} column {col+1}: expected {a!r}, found {b!r}")
        mode = {4: "cited static comparison", 10: "exploratory aggregate replay", 11: "exploratory aggregate replay"}.get(
            num, "public measurement replay")
        report["tables"][name] = {"rows": len(rows), "fields_including_labels": len(rows) * len(header),
                                 "mode": mode}
    report.update(status="failed" if mismatches else "passed", differences=mismatches)
    (dest / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if mismatches:
        raise ValueError(f"{len(mismatches)} manuscript table differences: {mismatches[:3]}")
    print(f"Verified {len(tables)} manuscript tables ({sum(map(len, tables.values()))} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output", help="output directory used by reproduce.py")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="directory containing the pinned public measurements")
    options = parser.parse_args()
    build(options.out, options.data)
