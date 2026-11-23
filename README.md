# Tail-CRC

**Decoupling Solving from Abstention: Single-Rollout Conformal Risk Control for Hard-Budget LLM Reasoning**

Qianyi Chen and Bo Li

[![Manuscript PDF](https://img.shields.io/badge/manuscript-PDF-943d2b)](Tail-CRC.pdf)
![arXiv pending](https://img.shields.io/badge/arXiv-pending-lightgrey)
[![Hugging Face observations](https://img.shields.io/badge/Hugging%20Face-observations-e6ad27)](https://huggingface.co/datasets/Cqyiii/tail-crc-observations)
[![License Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-31746b)](LICENSE)

Tail-CRC trains a solver to finish a mathematical answer under a hard token budget, then uses token surprisal from that same rollout to decide whether to submit it. A held-out family set calibrates the decision for a chosen marginal wrong-submission risk.

[Reproduce the paper](#reproduce-the-paper) · [Try the decision rule](#quickstart) · [Results](#results) · [Solver weights](ADAPTERS.md) · [Citation](#citation)

![The solver is trained with binary rewards, frozen, and scored on the sampled answer span. Calibration families determine the submission cutoff.](assets/fig_method_overview.png)

[Method overview · PDF](assets/fig_method_overview.pdf) · [SVG](assets/fig_method_overview.svg)

## Quickstart

From a clone of this repository, this example runs on standard Python. The score averages the largest ⌈L/10⌉ token surprises anywhere in the L-token answer span. Calibration uses one labelled rollout per family, including families that yielded no valid answer.

```python
from math import inf
from tail_crc import crc_threshold, tail10_nll

# One CAL family per row, all from the same frozen solver and seed.
cal = [("W", 0.3), ("R", 0.1), ("W", 0.7), ("U", inf)]
cut = crc_threshold(cal, "0.4")  # j = floor(5 * 0.4) = 2

# Sampled answer-span log probabilities after the first close and before EOS.
# Remove prefill and injected delimiters/newlines before calling tail10_nll.
for logp, valid in [([-0.1, -0.2], True), ([-0.7], True), ([], False)]:
    score = tail10_nll(logp, valid=valid)
    submit = valid and score < cut
    print(f"score={score:g} cutoff={cut:g} submit={submit}")

assert cut == 0.7 and not (0.7 < cut)  # reject cutoff ties
assert crc_threshold(cal, "0.1") == -inf  # j = 0
assert crc_threshold(cal, "0.8") == inf   # fewer than j wrong answers
```

The strict cutoff rejects every tie. Under exchangeable families and a solver, decoder, verifier and score fixed before calibration, Theorem 1 controls the expected marginal probability of a wrong submission on a new family. A realized test-panel rate can exceed the target. [Reference implementation](tail_crc/__init__.py)

## Results

![Correct, wrong and unsubmitted outcomes under a 4K cap across two model sizes and domains.](assets/fig1_three_state.png)

[Hard-budget outcomes · PDF](assets/fig1_three_state.pdf) · [SVG](assets/fig1_three_state.svg)

Correct / wrong submissions below are percentages of all TEST families. Each model uses same-domain calibration at α = 6.5% and averages the two solver seeds within each family.

| Domain | Model | Cap | Binary RL | Tail-CRC |
|---|---|---:|---:|---:|
| DAPO | 4B | 4K | 47.9 / 5.0 | 47.9 / 5.0 |
| DAPO | 4B | 8K | 57.6 / 6.3 | 57.6 / 6.3 |
| DAPO | 8B | 4K | 41.4 / 18.9 | 33.9 / 7.1 |
| DAPO | 8B | 8K | 57.5 / 26.4 | 39.1 / 5.9 |
| Omni-MATH | 4B | 4K | 25.3 / 9.8 | 17.8 / 3.7 |
| Omni-MATH | 4B | 8K | 30.5 / 12.8 | 24.3 / 5.8 |
| Omni-MATH | 8B | 4K | 29.3 / 30.3 | 16.7 / 5.8 |
| Omni-MATH | 8B | 8K | 38.8 / 41.8 | 18.0 / 6.5 |

On 4B DAPO both fitted policies accept every valid answer. On 8B DAPO at 4K, wrong submissions reach 7.1% on TEST. On Omni-MATH, target-domain calibration lowers wrong submissions at a cost in correct submissions. Measured token savings compare **Binary RL with the base model under the same budget forcing**. Withholding an already generated answer retains its generation cost. [Omni-MATH comparison · SVG](assets/omni-calibration.svg) · [PDF](assets/omni-calibration.pdf)

## Reproduce the paper

Use Python 3.11. The paper build runs on CPU. The first run downloads eight checksum-pinned numerical inputs from Hugging Face.

```sh
git clone --branch v1.1.1 https://github.com/Cqyiiii/tail-crc.git
cd tail-crc
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-paper.txt
make paper
```

On macOS, older Conda-built Python interpreters can report the system version as 10.16 to pip. If a compatible SciPy wheel is missing on macOS 14 or later, run `SYSTEM_VERSION_COMPAT=0 python -m pip install -r requirements-paper.txt`.

- `output/paper/tables/` contains 12 paper tables and a field-level verification report. Nine tables derive from public measurements, one compares published method definitions, and two reconstruct exploratory A3 results from recorded aggregates. [Table guide](paper/TABLES.md)
- `output/paper/figures/` contains eight figures in PDF, SVG and PNG. The case illustration displays sampled token counts in place of verbatim outputs. Training-curve active-group fractions are supplied as recorded step aggregates. [Supplementary inputs](paper/inputs/README.md)
- `make theory` runs the reference-rule tests and the CRC, score and RL numerical checks. These check enumerations, boundary cases and numerical examples from the manuscript.
- `make reproduce` rebuilds the underlying six evidence tables and five diagnostic plots, and checks 43,185 fields. To replay offline, prepopulate `data/` with the eight files in [the manifest](data-files.json). The loader verifies their sizes and SHA-256 values.

This release starts from recorded grades, scores and costs. The anonymized measurements support paired family-level analyses. Independent grading from raw answers, fresh model generation and new GPU timing require additional inputs. Four DEV parity means and eight CPU timing quantiles survive as recorded summaries. The 484-family reserve remains sealed.

## Repository guide

| Directory | Contents |
|---|---|
| [tail_crc](tail_crc/) and [examples](examples/) | Standard-library score, CRC cutoff and runnable example |
| [replay](replay/) and [validation](validation/) | Paired estimates, calibration, score and systems analyses, frozen references |
| [paper](paper/) and [theory](theory/) | Publication tables, figures, numerical checks and scoped inputs |
| [tools](tools/) | README chart, model conversion and author-side observation export |
| [release.json](release.json) and [data dictionary](DATA_CARD.md) | Pinned Hub revisions, data units and field definitions |

## Solver weights

Four public LoRA adapters cover Binary RL at both sizes and both seeds. Each repository provides the original factors, PEFT layout, checksums and base revision. A submission threshold is fitted separately for each calibration setting. [Model links, HF/vLLM loading and 4B reconstruction](ADAPTERS.md)

## Paper names and archived identifiers

| Paper name | Archived record |
|---|---|
| Native | `Native/base` |
| Budget forcing | `ExtBF/base` (decoder `BF` in observations) |
| Answer-SFT | 4B `E1_41`, 8B `M5_41` |
| Binary RL | `G0_41`, `G0_73` (pooled results average seeds within each family) |
| Binary RL with native decoding | `Native/G0_41` |
| Penalty RL | `G1_41` (wrong-answer reward −1) |
| Mild-Penalty RL | `S3B` (wrong-answer reward −0.25) |
| Explicit-Abstain RL | `S3A` (abstention action and wrong-answer reward −1) |
| Tail-CRC | Binary RL, `tail10_nll`, one CRC fit per seed |
| Tail / Box / Mean / Sum NLL | `tail10_nll`, `box_sum_nll`, `ans_mean_nll`, `ans_sum_nll` |
| Tail entropy | `tail_H` |
| Source / target calibration | Track A (DAPO cuts) / Track B (natural Omni CAL300 cuts) |
| DAPO source / independent Omni study | `pc_crc_confirm_2a_v1` / `pc_crc_ood_confirm_v1` |

## Citation

```bibtex
@misc{chenli2026tailcrc,
  title  = {Decoupling Solving from Abstention: Single-Rollout Conformal Risk Control for Hard-Budget {LLM} Reasoning},
  author = {Qianyi Chen and Bo Li},
  year   = {2026},
  note   = {Preprint}
}
```

The code is [Apache-2.0](LICENSE). The numerical observations and validation references use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the paper and record the release or full revisions used. [Software citation](CITATION.cff) · [Data and licenses](https://huggingface.co/datasets/Cqyiii/tail-crc-observations)
