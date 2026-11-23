"""Reproduce the tables and figures on CPU using only the included measurements."""
from pathlib import Path
import argparse
import gzip
import json
import os
import tempfile
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"output")
    options=parser.parse_args()
    out=options.out.resolve()
    data=ROOT/"data"
    if out in (ROOT,data,ROOT/"validation") or out.is_relative_to(data) or out.is_relative_to(ROOT/"validation"):
        parser.error("Choose a separate output directory.")
    out.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix="pc-crc-mpl-") as cache:
        os.environ["MPLCONFIGDIR"]=cache
        for key in ("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS"):
            os.environ[key]="1"
        from replay import main_panels,uncertainty,systems,figure_data,plots
        with gzip.open(data/"observations.jsonl.gz","rt") as stream:
            rows=[json.loads(line) for line in stream]
        assert len(rows)==42000 and len({r["family"] for r in rows})==1800
        start=time.perf_counter()
        main_panels.run(rows,out)
        uncertainty.run(rows,out)
        points=figure_data.read_csv(out/"main_points.csv")
        entropy=figure_data.read_csv(out/"whitebox_points.csv")
        box=[r for r in points if r["filter_rule"].startswith("crc_box_")]
        pooled=[r for r in points if not r["filter_rule"].startswith("crc_box_")]
        assert list(map(len,(pooled,entropy,box)))==[132,16,24]
        figure_data.save_csv(out/"table1_dapo_omni_main_results.csv",pooled+entropy+box)
        systems.run(data,out)
        figure_data.run(rows,data,out)
        plots.run(out)
        print(f"6 tables and 5 figures rebuilt in {time.perf_counter()-start:.1f} seconds")


if __name__=="__main__":
    main()
