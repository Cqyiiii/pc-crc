"""Render manuscript Figures 1–8 as PNG, PDF, and SVG after running reproduce.py."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import figures_empirical, figures_theory


INPUTS = Path(__file__).resolve().parent / "inputs"
FIGURES = {
    "fig1_three_state": "Three-state outcomes under a 4K budget",
    "fig_method_overview": "Training, scoring, calibration and deployment",
    "fig_theory_intuition": "Attempt starvation and two observed score decisions",
    "fig2_risk_coverage": "Eight-billion-parameter risk–coverage sweeps",
    "fig3_score_anatomy": "Wrong-answer discrimination by uncertainty score",
    "fig4_calibration": "Calibration domain, certificates and calibration size",
    "figA_penalty_barrier": "Attempt–solver phase boundary and trajectories",
    "figA_training_curves": "Eight-billion-parameter training trajectories",
}


def render(out):
    directory = out / "paper" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    written = set()

    def save(fig, name, *, dpi=220, bbox_inches="tight", pad_inches=None):
        if pad_inches is None:
            pad_inches = 0.08 if name in ("fig1_three_state", "fig3_score_anatomy") else 0.03
        common = dict(bbox_inches=bbox_inches, pad_inches=pad_inches, facecolor="white")
        for extension, metadata in [
            ("pdf", {"Title": FIGURES[name], "CreationDate": None, "ModDate": None}),
            ("png", {"Title": FIGURES[name]}),
            ("svg", {"Title": FIGURES[name], "Date": None}),
        ]:
            fig.savefig(directory / f"{name}.{extension}", dpi=dpi, metadata=metadata, **common)
        plt.close(fig)
        written.add(name)

    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "tail-crc-paper-figures",
    }
    with plt.rc_context(style):
        figures_empirical.render(out, INPUTS, save)
        with plt.rc_context({"font.size": 12}):
            figures_theory.method_overview(save)
            figures_theory.theory_intuition(
                figures_theory.load_cases(INPUTS / "qualitative_cases.json"), save
            )
        with plt.rc_context({"axes.titlesize": 11, "axes.linewidth": 0.6}):
            figures_theory.penalty_barrier(save)
    assert written == FIGURES.keys()
    print(f"Rendered {len(written)} paper figures in PNG, PDF and SVG in {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Replay output directory containing the regenerated CSVs and frontier.json",
    )
    render(parser.parse_args().out)
