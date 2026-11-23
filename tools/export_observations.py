"""Author-side export from released study artifacts. No text, receipts or service handles are inputs."""
import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def records(path):
    with path.open(encoding="utf-8") as inp:
        for line in inp:
            yield json.loads(line)


def dump_lines(path, rows):
    with path.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as dst:
        for row in rows:
            dst.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())


def export(project, destination):
    destination.mkdir(parents=True, exist_ok=True)
    confirm = project / "pc_crc_confirm_2a_v1"
    ood = project / "pc_crc_ood_confirm_v1"
    final = project / "paper_final_supplement_v1"
    pilot = project / "grpo_bf_pilot_v1"
    assert (confirm / "manifests/outcome_release.json").is_file()
    assert (ood / "manifests/test_release.json").is_file()
    assert all((final / f"audit/{stage}_complete.json").is_file() for stage in ("S2", "S3A", "S3B"))

    # Never materialize a candidate roster. Derive all public IDs from the scored records.
    features = {}
    for row in records(project / "pc_crc_postmortem_v1/features.jsonl"):
        k = (row["panel"], row["size"], row["arm"], row["budget"], row["family"])
        assert k not in features
        features[k] = row
    white = {}
    for path in sorted((project / "pc_crc_a100_supplement_v1/results").glob("whitebox_*.jsonl")):
        for row in records(path):
            k = tuple(row[f] for f in ("domain", "phase", "size", "arm", "budget", "family"))
            assert k not in white
            white[k] = row
    assert len(features) == 20000 and len(white) == 28800
    panel_rows = []
    ids = defaultdict(set)
    obs_keys = set()
    scores = ("box_sum_nll", "ans_sum_nll", "ans_mean_nll", "ans_len", "tail10_nll")
    def append(row, domain, phase, decoder, *, original=False):
        size, arm, budget, family = (row[f] for f in ("size", "arm", "budget", "family"))
        assert domain in ("DAPO", "Omni", "OmniEnriched") and phase in ("CAL", "TEST")
        assert size in ("4B", "8B") and arm in ("base", "G0_41", "G0_73", "E1_41", "M5_41", "G1_41", "S3A", "S3B")
        assert budget in (4096, 8192) and row["vs"] in ("R", "W", "U")
        assert row["strict"] in ("correct", "wrong", "truncated", "native_format", "explicit_abstain")
        assert row["u_reason"] in (None, "trunc", "abstain", "alternate_eos", "eos_no_box", "other")
        assert type(row["natural_close"]) is bool and type(row["output_tokens"]) is int and row["usd"] >= 0
        v = {"domain":domain, "phase":phase, "decode":decoder, "family":family, **{k:row[k] for k in
             ("size","arm","budget","vs","strict","u_reason","natural_close","output_tokens","usd")}}
        if "ref_is_bare_int" in row:
            assert type(row["ref_is_bare_int"]) is bool
            v["ref_is_bare_int"] = row["ref_is_bare_int"]
        if original:
            original_box = row.get("box_sum_nll")
            if original_box is not None:
                assert math.isfinite(original_box)
                v["box_sum_nll_2a"] = original_box
            key = ("OOD" if domain == "OmniEnriched" else phase,size,arm,budget,family)
            add = features.pop(key)
            assert add["vs"] == row["vs"]
            row = {**row, **add}
        for field in scores:
            if row.get(field) is not None:
                assert type(row[field]) in (int, float) and math.isfinite(row[field])
                v[field] = row[field]
        k = (domain,phase,size,arm,budget,family)
        if decoder == "BF" and domain != "OmniEnriched" and arm not in ("S3A","S3B"):
            add = white.pop(k)
            assert add["vs"] == v["vs"]
            v["wb_rank"] = add["rank"]
            assert type(v["wb_rank"]) is int
            for name in ("tail_H","tail_margin","receipt_tail"):
                if v["vs"] != "U":
                    assert math.isfinite(add[name])
                    v[name] = add[name]
        assert (*k,decoder) not in obs_keys
        obs_keys.add((*k,decoder))
        panel_rows.append(v)
        ids[domain,phase].add(family)

    for phase in ("CAL","TEST","OOD"):
        for src in sorted((confirm/"eval"/phase).glob("*/scores.jsonl")):
            for row in records(src):
                assert row["panel"] == phase
                append(row,"OmniEnriched" if phase=="OOD" else "DAPO","TEST" if phase=="OOD" else phase,"BF",original=True)
    for phase in ("CAL","TEST"):
        for sz in ("4B","8B"):
            for row in records(ood/f"eval/{sz}/{phase}_scores.jsonl"):
                assert row["phase"]==phase and row["size"]==sz
                append(row,"Omni",phase,"BF")
    for stage in ("S2","S3A","S3B"):
        for path in sorted((final/"eval"/stage).glob("*/scores.jsonl")):
            for row in records(path):
                append(row,row["domain"],row["phase"],"Native" if stage=="S2" else "BF")
    assert not white and not features
    expected = {("DAPO", "CAL"):500, ("DAPO","TEST"):500, ("Omni","CAL"):300, ("Omni","TEST"):300, ("OmniEnriched","TEST"):200}
    assert {k:len(v) for k,v in ids.items()} == expected
    all_ids = sorted(set.union(*ids.values()))
    assert len(all_ids) == sum(expected.values())
    aliases = {f:f"F{i:04d}" for i,f in enumerate(all_ids,1)}
    for row in panel_rows:
        row["family"] = aliases[row["family"]]
    panel_rows.sort(key=lambda r:tuple(r[k] for k in ("domain","phase","decode","size","arm","budget","family")))
    dump_lines(destination/"observations.jsonl.gz",panel_rows)

    # Recorded step summaries are sufficient to regenerate the training figure.
    paths = {"4B_G0_41":confirm/"train/4B_41","4B_G0_73":confirm/"train/4B_73",
             "8B_G0_41":pilot/"train/G0","8B_G0_73":confirm/"train/8B_73","8B_G1_41":pilot/"train/G1",
             "S3A":final/"train/S3A","S3B":final/"train/S3B"}
    steps=[]
    for arm,folder in paths.items():
        for i in range(1,51):
            v=json.loads((folder/f"step{i:03d}/done.json").read_text())
            m=v["metrics"]
            assert m["n"]==128 and sum(m["counts"].values())==128
            steps.append(dict(arm=arm,step=i,n=128,r=m["counts"].get("R",0),w=m["counts"].get("W",0),
                         trunc=m["u_reasons"].get("trunc",0),generated_sum=round(128*m["mean_generated"])))
    dump_lines(destination/"training_steps.jsonl.gz",steps)
    print(len(panel_rows),"observations;",len(all_ids),"families;",len(steps),"optimizer steps")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    export(args.project,args.destination)
