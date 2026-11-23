"""Render the README comparison from the CPU replay's operating points."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]

def render(results, output):
    with results.open(newline="") as source:
        rows = {(r["filter_rule"], int(r["budget"])): r for r in csv.DictReader(source)
                if (r["domain"], r["size"], r["arm"], r["decode"]) == ("Omni", "8B", "G0_pooled", "ExtBF")
                and r["filter_rule"] in ("raw", "crc_tail10_0.065")}
    assert set(rows) == {(rule, cap) for rule in ("raw", "crc_tail10_0.065") for cap in (4096, 8192)}
    ink, muted, teal = "#233942", "#657680", "#087668"
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 2.95), layout=None)
    fig.subplots_adjust(left=.065, right=.965, bottom=.20, top=.72, wspace=.22)
    for ax, metric, title in zip(axes, ("R_pct", "W_pct"), ("Correct submissions", "Wrong submissions")):
        ax.set_title(title, loc="left", color=ink, fontsize=14, weight="bold", pad=10)
        for y, cap in [(1, 4096), (0, 8192)]:
            raw = float(rows["raw", cap][metric])
            selected = float(rows["crc_tail10_0.065", cap][metric])
            ax.plot([selected, raw], [y, y], color="#bfcecc", linewidth=2.5, zorder=1)
            ax.scatter([raw], [y], s=62, color="white", edgecolors=muted, linewidths=1.8, zorder=3)
            ax.scatter([selected], [y], s=68, color=teal, zorder=3)
            for value, color in [(raw, muted), (selected, teal)]:
                ax.annotate(f"{value:.1f}%", (value, y), xytext=(0,7), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, color=color)
        ax.set(xlim=(0, 46), ylim=(-.35, 1.57), yticks=[0,1], yticklabels=["8K","4K"], xticks=[0,10,20,30,40])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="both", length=0, colors=muted, labelsize=9)
        ax.grid(axis="x", color="#e8eeee", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_visible(False)
    markers = [Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=muted),
               Line2D([], [], ls="", marker="o", color=teal)]
    fig.legend(markers, ["Raw solver", "Tail-CRC"], ncol=2, loc="upper right",
               bbox_to_anchor=(.97,1), frameon=False, labelcolor=ink, fontsize=10)
    fig.text(.065,.035,"Share of all test families",fontsize=10,color=muted)
    fig.savefig(output,dpi=145,facecolor="white",bbox_inches="tight",pad_inches=.18)
    plt.close(fig)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"output/table1_dapo_omni_main_results.csv")
    p.add_argument("--output",type=Path,default=ROOT/"assets/omni-calibration.png")
    a=p.parse_args();render(a.results,a.output)
