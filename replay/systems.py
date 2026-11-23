"""Replay Table 6 from measurements on released DEV and TEST panels.

Wall seconds are the original warm-batch measurements. Pool elapsed seconds
and generated lengths across batches before dividing by families or seconds.
The vote pilot joins three draws by family. The binary agreement field was
computed by comparing both extracted answers as stripped, non-null strings.

The eight historical portability errors compare to the reference adapter on
one captured 32-token DEV path per model. Recompute four means from unordered
absolute token log-probability differences. Only aggregate MAEs were retained
for the other four comparisons (4B HF base, HF PEFT on both models, vLLM base).
The original CPU timer retained block quantiles only. Recompute the cutoff and
number accepted on the archived scores; retain the measured timing quantiles.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


FILENAME = "table6_a100_portability_latency_and_pilot.csv"
COLUMNS = ("category", "engine", "model", "arm", "decode_or_test", "n",
           "metric_1", "metric_2", "metric_3", "notes")
SIZES = ("4B", "8B")
G0 = "G0_41"


def records(data, name):
    with (data / name).open(newline="") as f:
        return list(csv.DictReader(f))


def portability_rows(data, measurements):
    errors = defaultdict(list)
    for r in records(data, "system_parity_errors.csv"):
        value = float(r["absolute_logprob_error"])
        assert math.isfinite(value) and value >= 0
        errors[r["engine"], r["model"], r["method"]].append(value)
    parity = {key: (len(values), fmean(values)) for key, values in errors.items()}
    for r in measurements["parity_aggregate_measurements"]:
        key = (r["engine"], r["model"], r["method"])
        assert key not in parity
        parity[key] = (r["n"], r["mae"])

    rows = []
    for size in SIZES:
        (n, baseline), (n_peft, peft), (n_merged, merged) = (
            parity["HF_PEFT", size, method] for method in ("base", "peft", "merged")
        )
        assert n == n_peft == n_merged == 32
        counts = measurements["adapter_counts"][size]
        note = f"{counts['loaded_peft_parameters']} tensors"
        if size == "4B":
            note = (f"{counts['exported_tensor_count']} exported LoRA tensors; "
                    f"{counts['loaded_peft_parameters']} PEFT parameters loaded; fused QKV + untied lm_head")
        rows.append(("Portability", "HF_PEFT", size, G0, "DEV_32tok", n,
                     f"base_MAE={baseline:.4f}", f"peft_MAE={peft:.4f}", f"merged_MAE={merged:.4f}", note))
    (n, baseline), (n_lora, lora) = (parity["vLLM", "8B", method] for method in ("base", "dynLoRA"))
    assert n == n_lora == 32
    version = ".".join(measurements["vllm_version"].split(".")[:2])
    rows.append(("Portability", f"vLLM_{version}", "8B", G0, "DEV_32tok_dynLoRA", n,
                 f"base_MAE={baseline:.4f}", f"dynLoRA_MAE={lora:.4f}", "",
                 "4B used PEFT-merged weights in vLLM"))
    return rows


def wall_clock_rows(data):
    counts = defaultdict(dict)
    for r in records(data, "system_generation_lengths.csv"):
        key = (r["model"], r["arm"], r["mode"], int(r["batch"]))
        family, generated = int(r["family"]), int(r["generated"])
        assert family not in counts[key] and generated > 0
        counts[key][family] = generated

    batches = defaultdict(dict)
    for r in records(data, "system_batch_times.csv"):
        key = (r["model"], r["arm"], r["mode"])
        batch = int(r["batch"])
        families = counts.pop((*key, batch))
        wall = float(r["wall_s"])
        assert len(families) == int(r["n"]) and wall > 0 and batch not in batches[key]
        batches[key][batch] = (families, wall)
    assert not counts

    rows = defaultdict(list)
    for weights in (G0, "base"):
        modes = ("native4096", "native8192", "ext4096", "ext8192")
        if weights == G0:
            modes = ("ext4096", "ext8192", "native4096", "native8192")
        for size in SIZES:
            for mode in modes:
                chunks = batches.pop((size, weights, mode))
                families, times = zip(*chunks.values())
                n = sum(len(f) for f in families)
                assert sorted(len(f) for f in families) == [4, 12, 34]
                assert n == len(set().union(*families)) == 50
                wall = math.fsum(times)
                rate = sum(sum(f.values()) for f in families) / wall
                note = "Base weights; 50 TEST families."
                if weights == G0:
                    note = "Same-adapter comparison. Offline filter feature extraction excluded."
                rows[weights].append(("WallClock", "vLLM_A100", size, weights, f"{mode}_DAPO", n,
                                      f"tok_s={rate:.4f}", f"batched_wall_s_per_problem={wall / n:.6f}", "", note))
    assert not batches
    return rows


def filter_rows(data, measurements):
    groups = defaultdict(list)
    for r in records(data, "system_filter_scores.csv"):
        score = float(r["box_sum_nll"]) if r["box_sum_nll"] else math.inf
        assert r["grade"] in ("R", "W", "U") and (score == math.inf) == (r["grade"] == "U")
        groups[r["model"], int(r["budget"]), r["phase"]].append((r["grade"], score))

    rows = []
    alpha = measurements["filter_alpha"]
    for size in SIZES:
        for budget in (4096, 8192):
            cal = groups.pop((size, budget, "CAL"))
            test = groups.pop((size, budget, "TEST"))
            assert len(cal) == len(test) == 500
            allowed = math.floor((len(cal) + 1) * alpha - 1 + 1e-10)
            wrong = sorted(score for grade, score in cal if grade == "W")
            cutoff = wrong[allowed] if allowed < len(wrong) else math.inf
            accepted = sum(score < cutoff for _, score in test)
            measured = measurements["cpu_quantiles"][f"{size}/{budget}"]
            rows.append(("CPU_Filter", "CPU", size, G0, f"DAPO_{budget}", len(test),
                         f"median_ns_prob={measured['median_ns_per_item']:.2f}",
                         f"P95_ns_prob={measured['p95_block_ns_per_item']:.2f}", f"accepted={accepted}",
                         "Threshold comparison on cached scores only; P95 across timing blocks."))
    assert not groups
    return rows


def voting_rows(data):
    groups = defaultdict(list)
    for r in records(data, "system_vote_families.csv"):
        groups[r["model"], r["domain"]].append(r)
    rows = []
    for size in SIZES:
        for domain in ("DAPO", "Omni"):
            group = groups.pop((size, domain))
            assert len(group) == len({r["family"] for r in group}) == 16
            assert all(r[k] in ("R", "W", "U") for r in group
                       for k in ("draw0_grade", "draw1_grade", "single_grade"))
            assert all(r["answers_agree"] in ("0", "1") for r in group)
            # With two draws the first draw wins every 1–1 tie. When the draws
            # agree, the first draw also represents the agreed answer.
            maj = Counter(r["draw0_grade"] for r in group)
            agree = Counter(r["draw0_grade"] for r in group if r["answers_agree"] == "1")
            single = Counter(r["single_grade"] for r in group)
            pair_tokens = sum(int(r["draw0_generated"]) + int(r["draw1_generated"]) for r in group)
            single_tokens = sum(int(r["single_generated"]) for r in group)
            change = (single_tokens / pair_tokens - 1) * 100
            rows.append(("Pilot_Maj2", "vLLM_A100", size, G0, f"{domain}_2x4K_vs_1x8K", len(group),
                         f"Maj2_RW=({maj['R']},{maj['W']})", f"Agree_RW=({agree['R']},{agree['W']})",
                         f"Single8K_RW=({single['R']},{single['W']})",
                         f"Tokens: {pair_tokens} vs {single_tokens} ({change:+.1f}%)"))
    assert not groups
    return rows


def run(data_dir, out_dir):
    data = Path(data_dir)
    destination = Path(out_dir)
    measurements = json.loads((data / "system_measurements.json").read_text())
    wall = wall_clock_rows(data)
    rows = [*portability_rows(data, measurements), *wall[G0], *filter_rows(data, measurements),
            *voting_rows(data), *wall["base"]]
    assert len(rows) == 27
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / FILENAME).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return destination / FILENAME
