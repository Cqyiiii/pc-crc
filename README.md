# PC-CRC · arXiv numerical reproduction package

Rebuild the six manuscript tables, four main figures and the training appendix on CPU. The inputs contain released family-level observations, training-step counts and systems measurements. Calibration is fit on CAL. The package includes the 8B failures at the original DAPO thresholds, unissued LTT certificates and the exploratory small-panel results.

## Run

Use Python 3.11 and an environment containing the pinned packages. From a clone of this repository:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

Installation needs access to a package index or a local wheelhouse. With matching wheels, install using `python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt`. The first replay fetches eight checksum-pinned numerical files from Hugging Face. Subsequent runs reuse verified local files and work offline. The numerical workflow has no GPU, service, account or API key dependency. On systems without Make run `python fetch_data.py`, `python -B reproduce.py` and `python -B verify.py` in sequence. Use `make reproduce OUT=/path/to/new-output` for another output directory. If a previous run failed, check the exit code before using any remaining files.

## Results

| Table / figure | What it estimates | Source in this package |
|---|---|---|
| 1 · operating points; Fig. 1 · hard-budget outcomes | R, W, U, net, coverage, strict R, termination, cost and output length; Native vs external budget forcing (ExtBF) | `main_panels.py` computes raw, within-domain tail10 and original 2A box policies. `uncertainty.py` fits entropy policies per seed. |
| 2 · paired effects and cost | Difference per family across arms, budgets and rules; original study-specific percentile intervals | 20,000 shared bootstrap draws per panel, using 20261121 for 2A and 20261122 for OOD and the supplement |
| 3 · penalty ablations; Fig. 2 · risk–coverage frontier | Two single-seed penalty alternatives, original strong penalty and G0; dense CAL-defined TEST sweep | `main_panels.py`; `frontier.json` retains every cut, including empty coverage |
| 4 · uncertainty; Fig. 3 · score behavior | Within-arm rank discrimination, entropy–receipt difference, paired CI, and accepted R/W; score distributions and target-reference strata | `uncertainty.py` uses the original A2 rank order and 2,000 bootstrap draws. Fig. 3 also uses per-family grades, reference-type flags and released scores. |
| 5 · transfer and recalibration; Fig. 4 · calibration size | Track A uses DAPO CAL; Track B uses independent Omni CAL. Unissued conditional certificates and sensitivity to calibration size are in companion CSVs. | `main_panels.py` and `figure_data.py`; sensitivity uses 2,000 common family resplits. |
| 6 · systems | DAPO throughput and time, DEV parity errors, cached-score CPU benchmark and the two-draw pilot | `systems.py` reconstructs statistics from recorded times, token counts, numerical residuals and deidentified voting data. Four DEV mean errors and eight CPU timing quantiles survive only as recorded summaries. |
| Fig. A · training | Five-step means for seven 50-step arms | `training_steps.jsonl.gz` contains observations per optimizer step; 128 rollouts per step. |

`output/` contains six CSVs, the point-level and family-bootstrap summaries, plot data, PNG/SVG files and `validation.json`. For publication, cite confidence intervals and study identities alongside the point values. The 4B adapter inventory has **498 exported tensors**. In the direct PEFT check, **402 parameter tensors loaded**. The 8B export has 506 tensors.

## Interpretation and checks

- R and W are correct and wrong valid submissions, U covers the remaining families. Coverage is R + W. Net is R − W. Conditional wrong risk is W / (R + W) where the denominator is positive. Costs attach to generated trajectories before filtering. Percentages in Table 1 use a different scale from fractional rates in Tables 2–5.
- G0 pooling averages observations of training seeds 41 and 73 within each family. A pooled observation can be fractional. The reported cost is the expectation of deploying one seed. Original 2A, natural target confirmation and fixed-panel supplements retain their own study and panel names.
- `validation/` contains independent numerical targets from the writing handoff and study records. They are read only by `verify.py`. Generation accepts no manuscript table as an input. Verification compares every row key and every field of all six tables, as well as frontier, sensitivity, certificates, training metrics, representation strata, distribution summaries and the paired white-box result. Ties, missing values, original rank order, money and timing units matter. New bootstrap experiments with different seeds target different exact intervals.
- This arXiv release covers numerical replay. The input begins with model outcomes and score features; it supports recalibrating and testing policies. Replicating the original generations, teacher-forced log probabilities or wall clock on another machine requires separate models, code and compute. The sealed 484-family Omni reserve has no entries in this package.

## Resources

[Evaluation records on Hugging Face](https://huggingface.co/datasets/Cqyiii/pc-crc-observations) · [Data schema and provenance](DATA_CARD.md). See `release.json` for the data revision and model inventory. Code is licensed under Apache-2.0. The derived numerical data are released under CC BY 4.0.

Optional model conversion uses `python tools/convert_adapter.py {4B,8B} ORIGINAL_DIR NEW_PEFT_DIR` with PyTorch and safetensors. The numerical replay does not import them.

## Pinned downloads

- [Numerical dataset · 8f0988432a75](https://huggingface.co/datasets/Cqyiii/pc-crc-observations/tree/8f0988432a75f9313c56c4a35cd881d00955e06b)
