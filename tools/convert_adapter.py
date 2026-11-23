"""Convert the published Tail-CRC original LoRA factors to Hugging Face PEFT layout."""
import argparse
import json
from pathlib import Path

BASES = {"4B": ("Qwen/Qwen3.5-4B","851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
         "8B": ("Qwen/Qwen3-8B","b968826d9c46dd6066d109eabc6255188de91218")}

def convert(size, original, output):
    import torch
    from safetensors.torch import load_file, save_file

    cfg = json.loads((original / "adapter_config.json").read_text())
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    assert rank == alpha == 32
    tensors = load_file(original / "adapter_model.safetensors", device="cpu")
    assert len(tensors) == {"4B": 498, "8B": 506}[size]
    pairs = {}
    for key, a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removesuffix(".lora_A.weight")
        b = tensors[module + ".lora_B.weight"]
        assert a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] == rank
        module = module.replace("model.unembed_tokens", "lm_head")
        if size == "4B":
            module = module.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")
        assert module not in pairs
        pairs[module] = a, b
    assert 2 * len(pairs) == len(tensors)
    fused = sorted(k.removesuffix(".in_proj_q") for k in pairs if k.endswith(".linear_attn.in_proj_q"))
    assert len(fused) == {"4B": 24, "8B": 0}[size]
    for module in fused:
        blocks = [pairs.pop(module + ".in_proj_" + key) for key in "qkv"]
        pairs[module + ".in_proj_qkv"] = torch.cat([a for a,b in blocks]), torch.block_diag(*[b for a,b in blocks])
    name, revision = BASES[size]
    config = dict(peft_type="LORA", task_type="CAUSAL_LM", r=rank, lora_alpha=alpha,
                  lora_dropout=0.0, bias="none", inference_mode=True, base_model_name_or_path=name,
                  revision=revision, target_modules=sorted({k.split(".")[-1] for k in pairs}),
                  rank_pattern={"in_proj_qkv": 3 * rank} if fused else {},
                  alpha_pattern={"in_proj_qkv": 3 * alpha} if fused else {})
    out = {}
    for name, (a, b) in pairs.items():
        out[name + ".lora_A.weight"] = a
        out[name + ".lora_B.weight"] = b
    assert len(out) == {"4B": 402, "8B": 506}[size]
    output.mkdir(exist_ok=False, parents=True)
    save_file(out, output / "adapter_model.safetensors", metadata={"format":"pt"})
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("size", choices=list(BASES))
    p.add_argument("original", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    convert(args.size, args.original, args.output)
