# Tail-CRC data dictionary

## Evaluation records

The evaluation file contains **42,000 deployment rows from 1,800 distinct released families**. DAPO contributes 500 CAL and 500 TEST families, the independent natural Omni-MATH panel contributes 300 CAL and 300 TEST, and the earlier enriched 2A Omni panel contributes 200 TEST. Multiple seeds, budgets, decoders and reanalyses share families. Source observations are from the original 2A and OOD confirmations, the final Native/penalty supplement and the A100 white-box audit. The 400-family training population is a separate resource; this package includes summarized training steps. The 484-family reserve remains sealed.

`observations.jsonl.gz` uses the unique key (domain, phase, family, size, arm, budget, decode). Anonymous family labels preserve the original lexical order over released families for reproducible paired bootstraps. They have no public inverse mapping. A2's phase-specific sampling rank is a separate integer used to reproduce that study's bootstrap order. Measurement tables for small systems experiments have their own local identifiers; a join to the main family label space is unnecessary.

| Field | Meaning |
|---|---|
| domain, phase, size, arm, budget, decode | DAPO / Omni / OmniEnriched, CAL or TEST, 4B or 8B, model or training arm, total token budget, historical BF or Native renderer |
| vs, strict, u_reason | Mathematical submission grade R/W/U, strict single-box grade or failure class, reason for U. A correct mathematical answer can have a different strict grade. |
| natural_close, output_tokens, usd | Observed trajectory closure, realized generated tokens and measured/attributed dollars per deployment under the historical accounting. Holding an answer after generation retains the cost. These columns describe the observed rollout, including when a policy later rejects it. |
| box_sum_nll, box_sum_nll_2a | Box negative log likelihood. Original 2A scores and later extraction remain separately addressable at floating-point precision. |
| tail10_nll, ans_sum_nll, ans_mean_nll, ans_len | Top-decile average token NLL in the generated answer, answer NLL sum and mean, answer token count. The components needed depend on the original study. |
| wb_rank, receipt_tail, tail_H, tail_margin | A2 rank, receipt tail score, teacher-forced full-vocabulary tail entropy and margin. Margin enters the AUROC risk ranking with reversed sign. U rows have no committed-answer score. |
| ref_is_bare_int | Natural Omni reference representation. This boolean describes the reference, separately from a prediction's format. |

An absent score means the feature was unavailable or the family made no valid submission. The replay implements the original eligibility and strict `score < cut` decision. Raw, unfiltered, pooled and conditional policies are named separately. Empty conditional metrics preserve an undefined denominator. Calibration-domain choice, train seed, panel and budget are part of the estimand.

| Domain | CAL families / deployments | TEST families / deployments |
|---|---:|---:|
| DAPO | 500 / 10,000 | 500 / 15,000 |
| Natural Omni-MATH | 300 / 6,000 | 300 / 9,000 |
| Earlier enriched Omni panel | — | 200 / 2,000 |

The 24 fields form a union across studies. `tail10_nll` is available on 19,688 valid outputs; `tail_H`, `tail_margin` and `receipt_tail` each have 14,846 values. Missing optional keys in JSON appear as nulls in tabular views.

## Files and provenance

| File under `data/` | Records | Fields and use |
|---|---:|---|
| `observations.jsonl.gz` | 42,000 | Deployment key, grades, per-answer scores and generation cost. Full dictionary below. |
| `training_steps.jsonl.gz` | 350 | `arm, step, n, r, w, trunc, generated_sum`. Seven arms, 50 steps each. |
| `system_generation_lengths.csv` | 800 | `model, arm, mode, batch, family, generated` |
| `system_batch_times.csv` | 48 | `model, arm, mode, batch, n, wall_s`. Pool tokens and wall time within configuration before dividing. |
| `system_filter_scores.csv` | 4,000 | `model, budget, phase, grade, box_sum_nll`. CPU timing applies to thresholding cached scores. |
| `system_parity_errors.csv` | 128 | `engine, model, method, absolute_logprob_error`. Some additional measurements survive as means only. |
| `system_vote_families.csv` | 64 | Model, domain, family, per-draw and single-draw grades and token counts, `answers_agree`. Four panels of 16 families. |
| `system_measurements.json` | One object | CPU protocol and quantiles, alpha, adapter tensor counts, recorded parity aggregates, vLLM version and units. |

The six locked numerical reference CSVs live in the [code repository](https://github.com/Cqyiiii/pc-crc/tree/v1.1.0/validation). They cover operating points, paired effects and cost, penalty ablations, score discrimination, transfer and conditional certificates, and systems measurements. The [table guide](https://github.com/Cqyiiii/pc-crc/blob/v1.1.0/paper/TABLES.md) maps those records to the 12 printed tables and identifies the static comparison and two A3 aggregate replays.

`paper/qualitative_cases.json` contains the 12 selected Appendix E case records. Each carries the paper case number, configuration, R/W/U grade, sampled token count, scores, cutoffs and decisions. Some cases also have a paired observation. The displayed cutoffs belong only to the primary arm. Null cutoffs with the unbounded flag represent positive infinity. Case selection was illustrative. Original problem text, raw outputs and family lookup keys are omitted. `paper/training_active_groups.csv` supplies the 150 step-level active-group fractions used in the training figure. [Field dictionary](paper/inputs/README.md)

## Upstream attribution

| Upstream source | Frozen revision | License information |
|---|---|---|
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) | `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0 |
| [Processed DAPO, English train](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed/tree/31dd309567e3da778038cc87d868b6097a3ccf68) | `31dd309567e3da778038cc87d868b6097a3ccf68` | No separate license tag at this revision. The [original DAPO card](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) lists Apache-2.0. |
| [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH/tree/40ba231d8f16e29ecd40e6407e2c8640145a8f62) | `40ba231d8f16e29ecd40e6407e2c8640145a8f62` | Apache-2.0 on the upstream card. Original contest material retains its attribution. |

Numerical records and evaluation summaries are released under CC BY 4.0. The companion code and four released solver adapters use Apache-2.0. The package contains derived observations; upstream question texts and base model weights are obtained from their respective sources.
