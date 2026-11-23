"""Calibration-size resplits, conditional certificates and quantitative figure inputs."""
import csv
import gzip
import json
import math
from collections import defaultdict
import numpy as np
from scipy.stats import beta, binom, rankdata

from tail_crc import _calibration_rank


def save_csv(path, rows):
    rows=list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        out=csv.DictWriter(stream,fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def certificates(groups):
    grids={"tail10_nll":[.5,.75,1,1.25,1.5,1.75,2,2.35,2.8,math.inf],
           "ans_mean_nll":[.06,.10,.14,.18,.22,.27,.33,.4,.5,math.inf]}
    results=[]
    for (domain,phase,size,arm,budget),cal in sorted(groups.items()):
        if phase!="CAL":
            continue
        test=groups[domain,"TEST",size,arm,budget]
        for score,grid in grids.items():
            c,t=([r.get(score,math.inf) for r in records] for records in (cal,test))
            cal_wrong=np.array([r["vs"]=="W" for r in cal])
            test_wrong=np.array([r["vs"]=="W" for r in test])
            for target in (.1,.15,.2,.3):
                selected=None
                for threshold in grid:
                    accepted=np.asarray(c)<threshold
                    m,w=int(accepted.sum()),int((accepted & cal_wrong).sum())
                    probability=float(binom.cdf(w,m,target))
                    if m and probability<=.01:
                        selected=(threshold,m,w,probability)
                threshold,m,w,probability=selected or (None,None,None,None)
                accepted_n=wrong_n=coverage=error=upper=None
                if selected:
                    fired=np.asarray(t)<threshold
                    accepted_n,wrong_n=int(fired.sum()),int((fired & test_wrong).sum())
                    coverage=accepted_n/len(test)
                    if accepted_n:
                        error=wrong_n/accepted_n
                        upper=1.0 if wrong_n==accepted_n else float(beta.ppf(.95,wrong_n+1,accepted_n-wrong_n))
                results.append(dict(domain=domain,size=size,budget=budget,arm=arm,score=score,target=target,
                    issued=bool(selected),threshold=threshold,cal_accepted=m,cal_wrong=w,cal_p=probability,
                    test_n=len(test),test_accepted=accepted_n,test_wrong=wrong_n,test_coverage=coverage,
                    test_error=error,test_upper95=upper))
    assert len(results)==288
    return results


def resplit_thresholds(wrong_scores, alpha):
    """Fit each resplit at once. Each CAL row uses +inf for every R or U."""
    j = _calibration_rank(wrong_scores.shape[1], alpha)
    if j == 0:
        return np.full(wrong_scores.shape[0], -np.inf)
    return np.partition(wrong_scores, j - 1, axis=1)[:, j - 1]


def sensitivity(groups):
    results=[]
    for domain,n,ns in (("DAPO",1000,(50,100,200,300,500)),("Omni",600,(50,100,200,300))):
        rng=np.random.default_rng(20261122)
        orders=np.array([rng.permutation(n) for _ in range(2000)])
        for size in ("4B","8B"):
            for budget in (4096,8192):
                risks=[]
                for arm in ("G0_41","G0_73"):
                    pair=groups[domain,"CAL",size,arm,budget]+groups[domain,"TEST",size,arm,budget]
                    pair=sorted(pair,key=lambda r:r["family"])
                    assert len({r["family"] for r in pair})==n
                    risks.append(np.array([r["tail10_nll"] if r["vs"]=="W" else np.inf for r in pair])[orders])
                for nc in ns:
                    for alpha in (.065,.1):
                        cuts=[resplit_thresholds(v[:,:nc],alpha) for v in risks]
                        seed_risks=[(v[:,nc:]<cut[:,None]).mean(1) for v,cut in zip(risks,cuts)]
                        wrong=np.mean(seed_risks,axis=0)
                        results.append(dict(domain=domain,size=size,budget=budget,ncal=nc,alpha=alpha,
                             mean=wrong.mean(),sd=wrong.std(ddof=1),p95=np.quantile(wrong,.95),within=(wrong<=alpha+1e-12).mean()))
    return results


def auc(rows,feature):
    vals=[r for r in rows if r["vs"]!="U"]
    rank=rankdata([r[feature] for r in vals])
    wrong=np.array([r["vs"]=="W" for r in vals])
    nw=int(wrong.sum())
    return (rank[wrong].sum()-nw*(nw+1)/2)/(nw*(len(rank)-nw)) if 0<nw<len(rank) else None


def run(rows,data,out):
    groups=defaultdict(list)
    for row in rows:
        if row["decode"]=="BF" and row["domain"] in ("DAPO","Omni") and row["arm"] not in ("S3A","S3B"):
            groups[tuple(row[k] for k in ("domain","phase","size","arm","budget"))].append(row)
    save_csv(out/"ltt_all.csv",certificates(groups))
    save_csv(out/"ncal_all.csv",sensitivity(groups))
    # Paired-seed stratified AUROC is the mean of two within-arm AUROCs.
    strata=[]
    for size in ("4B","8B"):
        for budget in (4096,8192):
            seeds=[groups["Omni","TEST",size,arm,budget] for arm in ("G0_41","G0_73")]
            for bare in (True,False):
                selected=[[r for r in seed if r["ref_is_bare_int"]==bare] for seed in seeds]
                assert {r["family"] for r in selected[0]}=={r["family"] for r in selected[1]}
                for feature in ("tail10_nll","box_sum_nll"):
                    v=[auc(seed,feature) for seed in selected]
                    strata.append(dict(size=size,budget=budget,representation="bare" if bare else "nonbare",
                        feature=feature,n_families=len(selected[0]),pooled_auc=np.mean(v) if None not in v else None))
    save_csv(out/"figure3_strata.csv",strata)
    quantiles=[]
    for domain in ("DAPO","Omni"):
        for grade in ("R","W"):
            for feature in ("box_sum_nll","tail10_nll"):
                sample=[r[feature] for a in ("G0_41","G0_73") for r in groups[domain,"TEST","8B",a,4096] if r["vs"]==grade]
                quantiles.append(dict(domain=domain,grade=grade,feature=feature,n=len(sample),
                        **dict(zip(("p10","median","p90"),np.quantile(sample,[.1,.5,.9])))))
    save_csv(out/"figure3_scores.csv",quantiles)
    with gzip.open(data/"training_steps.jsonl.gz","rt") as stream:
        steps=[json.loads(s) for s in stream]
    save_csv(out/"training_steps.csv",(dict(arm=s["arm"],step=s["step"],R=s["r"]/s["n"],W=s["w"]/s["n"],
              trunc=s["trunc"]/s["n"],mean_generated=s["generated_sum"]/s["n"]) for s in steps))
