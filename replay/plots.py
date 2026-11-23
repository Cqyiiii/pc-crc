"""Generate the four manuscript figures and the training appendix from replay outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from .figure_data import read_csv

GREEN,BLUE,RED,GRAY="#06745e","#3b5e91","#ab482c","#667787"
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":105,"savefig.dpi":145,
                     "font.size":9,"axes.labelsize":9,"svg.fonttype":"none","svg.hashsalt":"pc-crc-arxiv-v1"})

def save(fig,out,name):
    fig.savefig(out/(name+".png"),facecolor="white",metadata={"Software":"Tail-CRC replay"})
    fig.savefig(out/(name+".svg"),facecolor="white",metadata={"Date":None})
    plt.close(fig)

def hard_budget(out,points):
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.6),layout="constrained",sharey=True)
    fig.suptitle("Hard-budget outcomes at 4,096 tokens",fontweight="bold")
    for ax,(domain,size) in zip(axes.flat,[(d,s) for d in ("DAPO","Omni") for s in ("4B","8B")]):
        candidates=[r for r in points if (r["domain"],r["size"],r["budget"],r["filter_rule"])==(domain,size,"4096","raw")]
        keys=[("Native","base"),("ExtBF","base"),("ExtBF","G0_pooled")]
        v=[next(r for r in candidates if (r["decode"],r["arm"])==key) for key in keys]
        bottom=np.zeros(3)
        for field,label,color in (("R_pct","Correct",GREEN),("W_pct","Wrong",RED),("U_pct","Unsubmitted","#dce4e7")):
            heights=[float(r[field]) for r in v]
            ax.bar(range(3),heights,bottom=bottom,color=color,label=label,width=.68)
            bottom+=heights
        ax.set(title=f"{domain} · {size}",xticks=range(3),xticklabels=["Native base","ExtBF base","ExtBF G0"],ylim=(0,101))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    axes[0,0].legend(fontsize=7,loc="upper center",ncol=3,framealpha=1)
    save(fig,out,"fig1_hard_budget")

def frontier(out,points):
    dense=json.loads((out/"frontier.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(9,6.2),layout="constrained")
    fig.suptitle("8B selective risk across the calibrated threshold sweep",fontweight="bold")
    for ax,(dom,b) in zip(axes.flat,[(d,n) for d in ("DAPO","Omni") for n in (4096,8192)]):
        v=[r for r in dense if (r["domain"],r["budget"])==(dom,b) and r["conditional_error"] is not None]
        ax.plot([100*r["coverage"] for r in v],[100*r["conditional_error"] for r in v],color="#acb8bb",lw=.9,label="All thresholds")
        best=2.;edge=[]
        for r in sorted(v,key=lambda x:(-x["coverage"],x["conditional_error"])):
            if r["conditional_error"]<best:
                edge.append(r)
                best=r["conditional_error"]
        ax.plot([100*r["coverage"] for r in edge],[100*r["conditional_error"] for r in edge],color=GREEN,lw=1.2,label="Non-dominated")
        for arm,mark,col in (("G0_41","s",BLUE),("G1_41","^",RED),("S3A","x","#84366e"),("S3B","D","#715f24")):
            (r,)=[x for x in points if (x["domain"],x["size"],int(x["budget"]),x["decode"],x["arm"],x["filter_rule"])==(dom,"8B",b,"ExtBF",arm,"raw")]
            ax.scatter(float(r["Cov_pct"]),100*float(r["W_pct"])/float(r["Cov_pct"]),marker=mark,color=col,s=29,label=arm,zorder=3)
        ax.set(title=f"{dom} · {b//1024}K",xlabel="Coverage (%)",ylabel="Wrong among submissions (%)")
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7,ncol=2)
    save(fig,out,"fig2_risk_coverage")

def uncertainty(out):
    vals=read_csv(out/"table4_uncertainty_auroc_and_whitebox.csv")
    strata=read_csv(out/"figure3_strata.csv")
    scores=read_csv(out/"figure3_scores.csv")
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.1),layout="constrained")
    fig.suptitle("Answer-span uncertainty on the fixed test panels",fontweight="bold")
    for ax,domain in zip(axs[0],("DAPO","Omni")):
        v=sorted((r for r in vals if r["domain"]==domain),key=lambda r:(r["size"],r["arm"],r["budget"]))
        for i,r in enumerate(v):
            l,h=[float(r[k]) for k in ("diff_lower","diff_upper")]
            y=float(r["H_minus_receipt"])
            col=GREEN if l>0 else GRAY
            ax.plot([i,i],[l,h],color=col)
            ax.scatter(i,y,s=14,color=col)
        ax.axhline(0,color="black",lw=.7)
        ax.set(title=f"{domain} · entropy minus receipt AUROC",ylabel="Paired AUROC difference",
               xticks=range(len(v)),xticklabels=[r["size"]+" "+r["arm"]+" "+str(int(r["budget"])//1024)+"K" for r in v])
        ax.tick_params(axis="x",labelrotation=70,labelsize=6)
    ax=axs[1,0];v=[r for r in strata if r["size"]=="8B"]
    for offset,(feature,color,label) in enumerate((("box_sum_nll",GRAY,"Box NLL"),("tail10_nll",GREEN,"Tail-10% NLL"))):
        bars=[r for r in v if r["feature"]==feature]
        ax.bar(np.arange(len(bars))+(offset-.5)*.34,[float(r["pooled_auc"]) for r in bars],width=.34,color=color,label=label)
    ax.set(xticks=np.arange(len(bars)),xticklabels=[f'{r["representation"]}\n{int(r["budget"])//1024}K' for r in bars],
           ylim=(0,1),ylabel="AUROC",title="Omni · 8B G0, by reference form")
    ax.legend(fontsize=7)
    ax=axs[1,1];v=[r for r in scores if r["feature"]=="tail10_nll"]
    for i,r in enumerate(v):
        ax.plot([i,i],[float(r["p10"]),float(r["p90"])],color=GRAY,lw=2)
        ax.scatter(i,float(r["median"]),color=GREEN if r["grade"]=="R" else RED,s=26)
    ax.set(xticks=np.arange(len(v)),xticklabels=[r["domain"]+" "+r["grade"] for r in v],ylabel="Tail-10% NLL",
           title="8B G0 at 4K · median and 10th–90th percentiles")
    save(fig,out,"fig3_uncertainty")

def resplits(out):
    data=read_csv(out/"ncal_all.csv")
    fig,axes=plt.subplots(2,2,figsize=(8.4,5.7),layout="constrained")
    fig.suptitle("Test wrong-rate variability over 2,000 family resplits",fontweight="bold")
    for ax,(dom,sz) in zip(axes.flat,[(d,s) for d in ("DAPO","Omni") for s in ("4B","8B")]):
        for b,col in ((4096,GREEN),(8192,RED)):
            for a,style in ((.065,"-"),(.10,"--")):
                v=[r for r in data if (r["domain"],r["size"],int(r["budget"]),float(r["alpha"]))==(dom,sz,b,a)]
                ax.plot([int(r["ncal"]) for r in v],[100*float(r["sd"]) for r in v],style,marker="o",ms=3,color=col,label=f"{b//1024}K / {a:.1%}")
        ax.set(title=dom+" · "+sz,xlabel="Calibration families",ylabel="SD of test wrong rate (pp)")
        ax.grid(alpha=.16)
    axes[0,0].legend(fontsize=7,ncol=2)
    save(fig,out,"fig4_calibration_size")

def training(out):
    data=read_csv(out/"training_steps.csv")
    fig,axes=plt.subplots(2,2,figsize=(9,6.2),layout="constrained")
    fig.suptitle("Training under the frozen 4K budget",fontweight="bold")
    for ax,(field,label) in zip(axes.flat,(("R","Correct (%)"),("W","Wrong (%)"),("trunc","Truncated (%)"),("mean_generated","Generated tokens per rollout"))):
        for arm in dict.fromkeys(r["arm"] for r in data):
            v=[float(r[field]) for r in data if r["arm"]==arm]
            y=np.mean(np.reshape(v,(10,5)),axis=1)*(1 if field=="mean_generated" else 100)
            ax.plot(range(5,51,5),y,marker=".",lw=1,label=arm)
        ax.set(xlabel="Optimizer step (five-step means)",ylabel=label)
        ax.grid(alpha=.16)
    axes[0,0].legend(fontsize=7,ncol=2)
    save(fig,out,"figA_training")

def run(out):
    points=read_csv(out/"table1_dapo_omni_main_results.csv")
    hard_budget(out,points)
    frontier(out,points)
    uncertainty(out)
    resplits(out)
    training(out)
