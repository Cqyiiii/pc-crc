# Tail-CRC · numerical observations and provenance

## Populations and data grain

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

## Additional files

- `training_steps.jsonl.gz` has 350 rows, keyed by training arm and optimizer step; correct/wrong/truncated counts and total generated tokens are sufficient to compute the training figure. Original step metrics supply the acceptance reference.
- `system_generation_lengths.csv` and `system_batch_times.csv` contain 800 generated lengths and 48 measured batch durations. Pool tokens and elapsed time before forming throughput. Offline postprocessing time and entropy-scoring time are outside the wall-clock benchmark.
- `system_filter_scores.csv` has 4,000 CAL/TEST grade and scalar box scores for the CPU benchmark; the measured CPU operation compares cached scores with a threshold. `system_vote_families.csv` has 64 rows with per-draw grades, lengths and a boolean agreement indicator. Each pilot configuration has 16 families.
- `system_parity_errors.csv` stores 128 absolute log-probability residuals. `system_measurements.json` contains original CPU timing quantiles, the four parity measurements without stored residuals, tensor counts and benchmark configuration. This replay does not reconstruct an unarchived timing distribution.
- The author-side export in `tools/export_observations.py` documents the field projection from released experiment files, without an identity lookup table in the package. Replay itself only requires `data/`. The CPU score distributions in Figure 3 and figure layout are rendered for this release; the comparison with the paper concerns their scientific values.

## Upstream revisions

DAPO comes from the English training portion of the processed corpus at `31dd309567e3da778038cc87d868b6097a3ccf68`. The Omni-MATH source revision is `40ba231d8f16e29ecd40e6407e2c8640145a8f62`. The upstream model revisions were `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` for Qwen3.5-4B and `b968826d9c46dd6066d109eabc6255188de91218` for Qwen3-8B. This package distributes derived numerical data. Redistribution of original tasks, answers, model outputs or weights requires separate review under the applicable terms.

[Processed DAPO](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) · [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
