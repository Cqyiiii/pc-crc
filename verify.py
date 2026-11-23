"""Compare regenerated cells with manuscript reference values. Run after generation."""
from pathlib import Path
import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
KEYS={"table1":("domain","size","decode","arm","budget","filter_rule"),
      "table2":("study","panel","contrast","metric"),"table3":("cell",),
      "table4":("size","arm","domain","budget"),"table5":("size","budget","arm","track","score","alpha"),
      "table6":("category","engine","model","arm","decode_or_test"),
      "ltt":("domain","size","budget","arm","score","target"),
      "ncal":("domain","size","budget","ncal","alpha"),"training":("arm","step"),
      "figure3_strata":("size","budget","representation","feature"),"figure3_scores":("domain","grade","feature")}

def keyed(path,fields):
    with path.open(encoding="utf-8") as inp:
        reader=csv.DictReader(inp)
        rows={}
        for row in reader:
            key=tuple(row[k] for k in fields)
            assert key not in rows,(path.name,key)
            rows[key]=row
        return reader.fieldnames,rows


def compare(expected,observed,key,stats,errors):
    stats["cells"]+=1
    try:
        v=float(expected)
        w=float(observed)
    except (TypeError,ValueError):
        if expected!=observed:
            errors.append((key,expected,observed))
        return
    stats["numerical"]+=1
    if v==w:
        return
    delta=abs(w-v)
    if math.isfinite(delta):
        stats["max_abs"]=max(stats["max_abs"],delta)
    if not math.isclose(v,w,rel_tol=0,abs_tol=1e-12):
        errors.append((key,expected,observed))


def tree(expected,observed,path,stats,errors):
    if isinstance(expected,dict):
        assert expected.keys()==observed.keys(),path
        for k,v in expected.items():tree(v,observed[k],f"{path}/{k}",stats,errors)
    elif isinstance(expected,list):
        assert len(expected)==len(observed),path
        for i,(v,w) in enumerate(zip(expected,observed)):tree(v,w,f"{path}/{i}",stats,errors)
    else:compare(expected,observed,path,stats,errors)


def verify(out):
    val=ROOT/"validation";report={};errors=[]
    targets=sorted(val.glob("*.csv"))
    table_targets=[p for p in targets if p.name.startswith("table")]
    assert [p.name.split("_")[0] for p in table_targets]==[f"table{i}" for i in range(1,7)]
    assert {p.name for p in targets if p not in table_targets}=={"ltt_all.csv","ncal_all.csv","training_steps.csv","figure3_strata.csv","figure3_scores.csv"}
    for path in targets:
        prefix=path.name.split("_")[0] if not path.name.startswith("figure3_") else path.stem
        fields=KEYS[prefix];names,expected=keyed(path,fields)
        columns,actual=keyed(out/path.name,fields)
        assert names==columns and expected.keys()==actual.keys(),path.name
        stats=dict(rows=len(expected),cells=0,numerical=0,max_abs=0.0)
        for k,row in expected.items():
            for col,x in row.items():
                compare(x,actual[k][col],f"{path.name} {'/'.join(k)} {col}",stats,errors)
        report[path.name]=stats
    for ref,actual in (("whitebox_vs_native_paired.json","whitebox_pair.json"),("risk_coverage_frontier.json","frontier.json")):
        stats=dict(cells=0,numerical=0,max_abs=0.0)
        tree(json.loads((val/ref).read_text()),json.loads((out/actual).read_text()),ref,stats,errors)
        report[actual]=stats
    for fig in ("fig1_hard_budget","fig2_risk_coverage","fig3_uncertainty","fig4_calibration_size","figA_training"):
        svg=ET.parse(out/(fig+".svg")).getroot()
        assert svg.tag.endswith("svg") and len(list(svg.iter()))>70
        assert (out/(fig+".png")).read_bytes()[:8]==b"\x89PNG\r\n\x1a\n"
    report=dict(tolerance_abs=1e-12,comparisons=report,errors=errors)
    (out/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if errors:
        raise ValueError(f"{len(errors)} reference mismatches: {errors[:3]}")
    print(f"Verified {sum(v['cells'] for v in report['comparisons'].values()):,} fields plus 5 figure files in both formats.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"output")
    verify(parser.parse_args().out)
