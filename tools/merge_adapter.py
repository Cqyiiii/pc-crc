"""Merge a released Tail-CRC PEFT adapter into its local, sharded Qwen base on CPU."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

BASES = {
    "Qwen/Qwen3.5-4B": ("qwen3_5", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 201),
    "Qwen/Qwen3-8B": ("qwen3", "b968826d9c46dd6066d109eabc6255188de91218", 253),
}
# Ancillary files used by the two base-model loaders and tokenizers/processors.
ASSETS = (
    "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json", "LICENSE", "NOTICE",
)
HEAD = "lm_head.weight"
EMBED_4B = "model.language_model.embed_tokens.weight"


def merge(base, adapter, out):
    import torch
    from safetensors.torch import load_file, save_file

    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    cfg = json.loads((base / "config.json").read_text())
    idx = json.loads((base / "model.safetensors.index.json").read_text())
    pcfg = json.loads((adapter / "adapter_config.json").read_text())
    model_type, revision, target_count = BASES[pcfg["base_model_name_or_path"]]
    if cfg["model_type"] != model_type or pcfg.get("revision") != revision:
        raise ValueError("The base architecture and adapter revision must match the released configuration.")
    is_4b = model_type == "qwen3_5"
    pattern = {"in_proj_qkv": 96} if is_4b else {}
    if not (pcfg["peft_type"] == "LORA" and pcfg["bias"] == "none"
            and pcfg["r"] == pcfg["lora_alpha"] == 32
            and pcfg["rank_pattern"] == pcfg["alpha_pattern"] == pattern):
        raise ValueError("Expected the published rank-32 PEFT layout with its documented QKV overrides.")
    factors = load_file(adapter / "adapter_model.safetensors", device="cpu")
    deltas = {}
    for key, a in factors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removesuffix(".lora_A.weight")
        b = factors[module + ".lora_B.weight"]
        if not module.startswith("base_model.model."):
            raise ValueError(f"Unexpected PEFT module name: {module}")
        weight = module.removeprefix("base_model.model.") + ".weight"
        suffix = module.rsplit(".", 1)[-1]
        rank = pcfg["rank_pattern"].get(suffix, pcfg["r"])
        alpha = pcfg["alpha_pattern"].get(suffix, pcfg["lora_alpha"])
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"Factor shapes disagree with the configured rank for {module}.")
        deltas[weight] = (a, b, alpha / rank)
    if len(factors) != 2 * target_count or len(deltas) != target_count or HEAD not in deltas:
        raise ValueError("Incomplete adapter. Supply the root PEFT files for a released solver.")
    weights = idx["weight_map"]
    original = weights.copy()
    if is_4b:
        if (HEAD in weights or EMBED_4B in deltas or not cfg["text_config"]["tie_word_embeddings"]
                or not cfg["tie_word_embeddings"]):
            raise ValueError("Expected the pinned 4B base with a tied output head.")
        weights[HEAD] = original[EMBED_4B]
        cfg["tie_word_embeddings"] = False
        cfg["text_config"]["tie_word_embeddings"] = False
    missing = deltas.keys() - weights.keys()
    if missing:
        raise ValueError(f"Adapter targets absent from the base index: {sorted(missing)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    with tempfile.TemporaryDirectory(prefix=".tail-crc-merge-", dir=out.parent) as work:
        dest = Path(work) / "model"
        dest.mkdir()
        written = 0
        for shard in sorted(set(original.values())):
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError(f"Unexpected shard filename: {shard}")
            tensors = load_file(base / shard, device="cpu")
            if tensors.keys() != {k for k, filename in original.items() if filename == shard}:
                raise ValueError(f"Tensor names disagree with the base index in {shard}.")
            if is_4b and EMBED_4B in tensors:
                tensors[HEAD] = tensors[EMBED_4B].clone()
                idx["metadata"]["total_size"] += tensors[HEAD].nbytes
            for weight in sorted(deltas.keys() & tensors.keys()):
                a, b, scale = deltas[weight]
                value = tensors[weight]
                if value.shape != (b.shape[0], a.shape[1]) or not value.is_floating_point():
                    raise ValueError(f"Incompatible base tensor for {weight}")
                a = a.float()
                for start in range(0, b.shape[0], 2048):
                    part = slice(start, start + 2048)
                    updated = value[part].float() + (b[part].float() @ a) * scale
                    value[part].copy_(updated.to(value.dtype))
                written += 1
            save_file(tensors, dest / shard, metadata={"format": "pt"})
            del tensors
        assert written == len(deltas)
        for name in ASSETS:
            if (base / name).is_file():
                shutil.copyfile(base / name, dest / name)
        (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (dest / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
        dest.rename(out)
    print(f"Merged {written} target modules into {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Pinned sharded base snapshot")
    parser.add_argument("--adapter", type=Path, required=True, help="Directory containing root PEFT files")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    merge(args.base, args.adapter, args.out)
