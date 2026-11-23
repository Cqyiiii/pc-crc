# Rebuilding the manuscript tables

From the repository root, run `make reproduce` and then `python -m paper.tables --out output`. The second command reads the replay outputs and the pinned measurements in `data/`. The twelve publication tables and `verification.json` appear in `output/paper/tables/`. Use `--data` for a different copy of the pinned measurements.

The CSVs follow the printed row order and precision. Rates labeled `pct` are percentages. Differences in `pp` are percentage points. Confidence interval endpoints occupy separate columns. The verifier compares column names, row labels, missing entries and formatted values against the independent transcription in `references/tables/`. These reference files are used only after generation. Table 4 additionally checks the five citation URLs against the manuscript bibliography.

| Manuscript table | Reproduction source | Interpretation |
|---|---|---|
| 1. Hard-cap outcomes | Replay evidence tables 1 and 5 | All TEST families contribute to the R/W rates, including U outcomes. Tail-CRC uses same-domain calibration at α = .065. Binary RL cost saving uses same-model, same-cap Budget forcing as its denominator. |
| 2. Reward ablations | Replay evidence table 3 | 8B at 4K. Binary RL pools seeds 41 and 73 within family. Each alternative uses seed 41. Net is R minus W after fitting each arm separately. |
| 3. Generation time | WallClock rows in replay evidence table 6, pooled from batch times and family counts | The denominator for each speed comparison is Native 8K with base weights on DAPO TEST50. Seconds per family measure warm, amortized batch time. Offline score extraction and queueing are outside the measurement. |
| 4. Calibrated decisions | Static, cited definitions in `table_inputs/method_comparison.csv` | The row wording and citation numbers come from Table 4. The final CSV column links each cited construction to its source. |
| 5. Raw solver contrasts | Replay evidence tables 1 and 2 | Budget-forced Binary RL versus same-cap base and Answer-SFT. DAPO uses the original source study and Omni uses the independent target confirmation. Costs represent a single deployed seed averaged over the two policies. |
| 6. Native decoding | Replay evidence tables 1 and 2 | Same-family Native seed-41 RL versus Native base, with 20,000 paired family-bootstrap resamples. |
| 7. Uncertainty | Replay evidence table 4 | AUROC among valid TEST submissions, with W positive. The entropy-minus-Tail-NLL intervals are pointwise paired-family intervals from up to 2,000 estimable resamples. |
| 8. Selective-error certificates | All 288 settings in `ltt_all.csv` | CAL issuance includes all candidate settings in its denominator. The two TEST columns summarize settings with a certificate and defined selective error. Every issued setting in the manuscript has a positive TEST acceptance count. DAPO is retrospective. Omni used independently registered grids. |
| 9. Calibration size | All 72 settings in `ncal_all.csv` | Each point uses 2,000 fixed-pool repartitions. Endpoint statistics use the smallest and largest CAL sets. The monotonicity flag uses every CAL size at full precision. |
| 10. Exploratory local training | 32 budget-forcing aggregate rows in `table_inputs/a3_final_tier.csv` | Correct/wrong counts use CAL16/TEST16. Qwen uses 12 updates and Llama uses 25. |
| 11. Exploratory Native decoding | 12 Llama Native aggregate rows in the same A3 file | Each row evaluates a completed 25-update local checkpoint on CAL16/TEST16. |
| 12. Exploratory two-draw comparison | `system_vote_families.csv`, 16 matched TEST families per model/domain | The first 4K draw breaks a tie. Agreement requires both answers to agree. Token totals sum both 4K draws or the single 8K draw. The 8B DAPO agreement row has 0 correct and 0 wrong submissions, so its conditional error is null in the verification report. |

## A3 aggregate provenance

The A3 file contains four local A100 study summaries joined by engine, model, arm, update tier, domain, decoder and cap. The training summary supplies updates, training rollouts and family exposures. Generation summaries supply raw TEST counts and panel sizes. The CRC summary supplies TEST counts after same-domain Tail-NLL calibration at α = .065. The mechanism contrasts supply the paired raw correct-rate difference and its pointwise 95% interval. The paired reference count equals the local raw correct count minus 16 times that archived difference. This reconstruction preserves the matched reference roster when a base evaluation used more families.

Qwen training used 192 rollouts over 48 family exposures per arm. DeepSeek-R1-Distill-Llama-8B used 400 over 100 exposures per arm. G0 rows compare with the same base. Other rows compare with local G0 seed 41 at the same tier. One family corresponds to 6.25 percentage points. The public aggregate projection replays the table arithmetic, counts and reported intervals. Fitting new thresholds or resampling the original A3 families would require the underlying pilot observations. These pilots were exploratory and do not enter the main confirmatory panels.
