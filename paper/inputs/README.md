# Supplementary inputs for the paper figures

Both files contain recorded numerical measurements and categorical labels under CC BY 4.0. The empirical renderers read the remaining figure values from the replay output directory.

| File | Unit | Fields and interpretation |
| --- | --- | --- |
| `qualitative_cases.json` | 12 selected Appendix E cases, two per qualitative category | `case` follows the manuscript order. `eligible_in_category` counts records that matched that category's selection rule. The selection was illustrative. It cannot provide population prevalence. |
| `training_active_groups.csv` | One training arm and optimizer step, 150 rows | The active-group fraction is the proportion of reward groups with nonconstant rewards at that step. There were 16 groups per step. The two 8B Binary RL seeds and one 8B Penalty RL seed each contribute 50 steps. |

Each case records the model size, evaluation domain, token budget, decoder, calibration domain, and marginal target `alpha`. The observed arm and any paired raw arm include the tolerant grade (`R`, `W`, or `U`), strict verifier label, sampled output token count, and three uncertainty scores. Scores can be null on invalid submissions. `box_cutoff` and `tail_cutoff` apply to the observed arm only. A null cutoff with the corresponding `*_cutoff_unbounded` flag set denotes positive infinity. The decision accepts valid submissions with a score strictly below the cutoff. A null `paired_raw` denotes a case without a paired response.

Figure 3 uses the numerical fields of Cases 5 and 10, replacing the illustrated answer strings with sampled output lengths. The published grade, score, thresholds, decisions and model phase boundary are preserved. Figure 8 joins the step aggregates to `training_steps.csv` by arm and step. The unsubmitted fraction is `1 − R − W`; all four time series use trailing means over at most five updates. An active group can contain only failures when failures have different rewards.
