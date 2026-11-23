"""Draw the README submission comparison as vector artwork from replayed operating points."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RULES = ("raw", "crc_tail10_0.065")
BUDGETS = (4096, 8192)


def render(results, output):
    with results.open(newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if (row["domain"], row["size"], row["arm"], row["decode"]) == ("Omni", "8B", "G0_pooled", "ExtBF")
                and row["filter_rule"] in RULES and int(row["budget"]) in BUDGETS]
    points = {(row["filter_rule"], int(row["budget"])): row for row in rows}
    assert len(rows) == len(points) == 4

    # The gap between budgets separates two comparisons made on the same test families.
    y = [3.4, 2.4, 1.0, 0.0]
    values = {rule: [float(points[rule, cap][metric]) for cap in BUDGETS for metric in ("R_pct", "W_pct")]
              for rule in RULES}
    colors = {"raw": "#536270", "crc_tail10_0.065": "#006d64"}
    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans",
                  rc={"axes.labelcolor": "#263946", "xtick.color": "#536270", "ytick.color": "#263946",
                      "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "tail-crc-omni"})
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    fig.subplots_adjust(left=.16, right=.975, top=.97, bottom=.31)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e6e8", linewidth=.65)
    ax.hlines(y, values[RULES[0]], values[RULES[1]], color="#adbfc0", linewidth=1.8, zorder=1)
    for rule, label, marker, face in [
        (RULES[0], "Raw solver", "o", "white"),
        (RULES[1], "Tail-CRC", "D", colors[RULES[1]]),
    ]:
        color = colors[rule]
        sns.scatterplot(x=values[rule], y=y, ax=ax, marker=marker, s=50, color=face,
                        edgecolor=color, linewidth=1.4, label=label, zorder=3)
        for rate, height in zip(values[rule], y):
            ax.annotate(f"{rate:.1f}%", (rate, height), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", color=color, fontsize=9)
    ax.set(xlim=(0, 45), ylim=(-.35, 4.0), xticks=range(0, 41, 10), yticks=y,
           yticklabels=["4K · correct", "4K · wrong", "8K · correct", "8K · wrong"],
           xlabel="Share of all test families", ylabel="")
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.tick_params(length=0, labelsize=9.5, pad=5)
    ax.xaxis.labelpad = 6
    sns.despine(ax=ax, left=True, bottom=True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.32), ncol=2, frameon=False,
              handletextpad=.3, columnspacing=1.6, fontsize=9.5)

    # An explicit filename still selects one format. A stem produces both vector files.
    paths = [output] if output.suffix else [output.with_suffix(ext) for ext in (".svg", ".pdf")]
    for path in paths:
        metadata = {".svg": {"Date": None}, ".pdf": {"CreationDate": None, "ModDate": None}}.get(path.suffix)
        fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=.12, metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "output/table1_dapo_omni_main_results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/omni-calibration",
                        help="Use a filename stem for both SVG and PDF, or a filename for one format.")
    args = parser.parse_args()
    render(args.results, args.output)
