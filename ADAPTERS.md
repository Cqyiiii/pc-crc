# Load and serve the released solvers

Four step-50 Binary RL adapters are public. The table pins the revisions containing the verified factor files. Each model repository contains the PEFT adapter at its root, the original inference factors in `original/`, and `verification.json` with file checksums. The parent studies archived seven adapters in total. The other three are the 4B Answer-SFT control (E1, seed 41), the 8B Answer-SFT control (M5, seed 41), and the 8B Penalty RL control (G1, seed 41). Their comparison results are in the numerical replay.

| Base | Seed | Adapter | Adapter revision |
| --- | ---: | --- | --- |
| Qwen3.5-4B | 41 | Cqyiii/PC-CRC-G0-4B-seed41 | [e8792cd4ecd58f53e6a4c97f8a4b8381ee60238c](https://huggingface.co/Cqyiii/PC-CRC-G0-4B-seed41/tree/e8792cd4ecd58f53e6a4c97f8a4b8381ee60238c) |
| Qwen3.5-4B | 73 | Cqyiii/PC-CRC-G0-4B-seed73 | [7c3303b38aa79c208a0656386b5c6edee892a7b2](https://huggingface.co/Cqyiii/PC-CRC-G0-4B-seed73/tree/7c3303b38aa79c208a0656386b5c6edee892a7b2) |
| Qwen3-8B | 41 | Cqyiii/PC-CRC-G0-8B-seed41 | [bc0e6714684e06b03102b0701dfee2710ef5b8f5](https://huggingface.co/Cqyiii/PC-CRC-G0-8B-seed41/tree/bc0e6714684e06b03102b0701dfee2710ef5b8f5) |
| Qwen3-8B | 73 | Cqyiii/PC-CRC-G0-8B-seed73 | [f8aa0372fa3fa5b9a1fe5fec5dfb9c0149c2201d](https://huggingface.co/Cqyiii/PC-CRC-G0-8B-seed73/tree/f8aa0372fa3fa5b9a1fe5fec5dfb9c0149c2201d) |

| Size | Pinned base | Original factor tensors | PEFT factor tensors | Target modules |
| --- | --- | ---: | ---: | ---: |
| 4B | [Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | 498 | 402 | 201 |
| 8B | [Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) | 506 | 506 | 253 |

## Load a PEFT solver

The historical A100 Transformers checks used PyTorch 2.8.0+cu128, Transformers 5.8.1, PEFT 0.19.1 and bfloat16. Model loading is optional for the offline numerical replay. The 4B checkpoint uses the multimodal model class even for text-only prompts.

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

# Choose one solver, here the 4B seed-41 example.
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    torch_dtype="auto",
)
solver = PeftModel.from_pretrained(
    base, "Cqyiii/PC-CRC-G0-4B-seed41", revision="e8792cd4ecd58f53e6a4c97f8a4b8381ee60238c",
).eval()
```

For 8B seed 41, use the causal language model class and the corresponding base and adapter.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218",
    torch_dtype="auto",
)
solver = PeftModel.from_pretrained(
    base, "Cqyiii/PC-CRC-G0-8B-seed41", revision="bc0e6714684e06b03102b0701dfee2710ef5b8f5",
).eval()
```

The two seeds at each size share one base revision. To use seed 73, substitute its model ID and adapter revision together. Each adapter provides one solver. Tail-CRC additionally uses the budget decoder, scores sampled tokens after the end-of-thinking delimiter, and fits a strict submission cutoff on the appropriate calibration families. See the [scoring and replay guide](README.md).

## Merge 4B for vLLM

The historical A100 run used a standalone merged checkpoint for 4B serving. The commands below require the Hugging Face CLI (`hf`), PyTorch and safetensors for the CPU merge, and a separate vLLM serving environment. Model repositories and their base-model licenses apply to the resulting files.

```sh
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir bases/4b
hf download Cqyiii/PC-CRC-G0-4B-seed41 adapter_config.json adapter_model.safetensors --revision e8792cd4ecd58f53e6a4c97f8a4b8381ee60238c --local-dir adapters/4b-41
python tools/merge_adapter.py --base bases/4b --adapter adapters/4b-41 --out merged/4b-41
vllm serve merged/4b-41
```

Give the merge tool a local sharded base snapshot from the specified revision and a fresh output path. It applies each scaled delta `(alpha / rank) * B @ A` in float32, then casts to the stored base weight dtype. It reads one base shard at a time and preserves tokenizer and processor files. On 4B it first copies the input embeddings to an independent output head, disables tying in both the top-level and text configurations, and applies the unembedding delta to the new head. The input embeddings retain their base values. The same command accepts an 8B sharded base and 8B PEFT adapter, where the base already stores the output head separately. [Merge implementation](tools/merge_adapter.py)

The root PEFT files are ready to load or merge. To reconstruct them from the two archived files in an adapter's `original/` directory, run `python tools/convert_adapter.py 4B adapters/4b-41/original rebuilt/4b-41` after downloading that subdirectory. Use `8B` with the corresponding 8B files. The conversion stacks the Q, K and V input factors and block-diagonalizes the output factors for 24 fused 4B linear-attention projections. Those projections use rank and alpha 96. The remaining projections use rank and alpha 32. [Factor conversion](tools/convert_adapter.py)

## Serve 8B with dynamic LoRA

The 8B seed-41 dynamic-adapter path was measured with vLLM 0.20.0. Requests to this server select the adapter by its registered model name, `tail-crc-8b-41`.

```sh
hf download Cqyiii/PC-CRC-G0-8B-seed41 adapter_config.json adapter_model.safetensors --revision bc0e6714684e06b03102b0701dfee2710ef5b8f5 --local-dir adapters/8b-41
vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --enable-lora --max-lora-rank 32 --lora-modules tail-crc-8b-41=adapters/8b-41
```

## Historical parity checks

Values are mean absolute differences in raw sampled-token log probability against the captured training-service path on 32 DEV tokens per adapter. Blank entries identify unmeasured routes. Dynamic 4B LoRA was unsupported in the historical serving environment. The supplement closed partially. New generations can differ with sampling, renderer and backend.

| Base | Seed | Transformers merged | Transformers PEFT | vLLM dynamic LoRA |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 41 | 0.00299518 | 0.00330210 | |
| Qwen3.5-4B | 73 | 0.00787374 | | |
| Qwen3-8B | 41 | 0.00216043 | 0.00116661 | 0.00073280 |
| Qwen3-8B | 73 | 0.00264194 | | |

The historical measurements appear in the pinned model repositories' `verification.json` files and the [system results](validation/table6_a100_portability_latency_and_pilot.csv). The local merger was also checked on synthetic two-shard checkpoints with all 201 or 253 target names and the published rank patterns. Reloaded tensors agreed with independently computed float64 reference updates after casting. The tests covered the independent 4B head, unchanged input embeddings, and correct shard-index byte counts.

For a local archive with the four model repositories stored under their short Hub names, verify all 16 original and PEFT files together with `sha256sum -c SHA256SUMS.txt`. Run the command in the common parent of the four directories. [Combined SHA-256 manifest](SHA256SUMS.txt)
