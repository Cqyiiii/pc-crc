"""Exact attempt–solver model and the paper's workflow and score decisions."""
import json
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit, logit


NAVY = "#275B8C"
TEAL = "#13867B"
RED = "#C75D4E"
GRAY = "#87939E"
GRID = "#D8DDE3"
LAMBDA = 1.0
KAPPA = 0.5
Q0 = 0.2
QC = LAMBDA / (1 + LAMBDA)
HORIZON = 40.0


def bernoulli_kl(target, initial):
    initial = np.asarray(initial)
    # The endpoint q0=0 is used only for the continuous boundary limit.
    with np.errstate(divide="ignore"):
        return target * np.log(target / initial) + (1 - target) * np.log((1 - target) / (1 - initial))


def critical_gate(q0, kappa):
    """Starvation is a0 <= critical_gate for 0 < q0 < qc."""
    return -np.expm1(-bernoulli_kl(QC, q0) / kappa)


def trajectory(a0):
    # Integrating logits maintains valid probabilities near 0 and 1.
    def rhs(_, state):
        a, q = expit(state)
        return [(1 + LAMBDA) * q - LAMBDA, KAPPA * (1 + LAMBDA) * a]

    time = np.linspace(0.0, HORIZON, 801)
    result = solve_ivp(rhs, (0.0, HORIZON), [logit(a0), logit(Q0)], t_eval=time, rtol=2e-11, atol=1e-12)
    if not result.success:
        raise RuntimeError(result.message)
    u, v = result.y
    a, q = expit(u), expit(v)
    invariant = QC * np.logaddexp(0, -v) + (1 - QC) * np.logaddexp(0, v) - KAPPA * np.logaddexp(0, u)
    if np.max(np.abs(invariant - invariant[0])) > 1e-8:
        raise AssertionError("The numerical path violates the exact invariant")
    return time, a, q


def penalty_barrier(save):
    a_critical = float(critical_gate(Q0, KAPPA))
    np.testing.assert_allclose(a_critical, 0.36, atol=2e-15, rtol=0)
    # Exact lower root for lambda=1, kappa=.5, q0=.2, a0=.25.
    q_starved = (1 - np.sqrt(11 / 75)) / 2
    curves = {a0: trajectory(a0) for a0 in (0.25, 0.50)}
    np.testing.assert_allclose(curves[0.25][2][-1], q_starved, atol=2e-7, rtol=0)
    if not (curves[0.25][1][-1] < 1e-6 and curves[0.50][1][-1] > 0.999 and curves[0.50][2][-1] > 0.999):
        raise AssertionError("Displayed model trajectories have unexpected limits")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.245, top=0.780, wspace=0.29)
    fig.suptitle(
        "Specified attempt–solver model",
        y=0.985,
        fontsize=12,
        fontweight="bold",
        color="#243746",
        va="top",
    )
    fig.text(
        0.5,
        0.915,
        "Exact model boundary and dynamics; no empirical fitting",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#566371",
    )

    # Panel (a): the recovery/starvation boundary is an analytic curve.
    ax = axes[0]
    q_grid = np.linspace(0.0, QC, 601)
    boundary = critical_gate(q_grid, KAPPA)
    ax.fill_between(q_grid, 0, boundary, color=RED, alpha=0.115, linewidth=0)
    ax.fill_between(q_grid, boundary, 1, color=TEAL, alpha=0.075, linewidth=0)
    ax.plot(q_grid, critical_gate(q_grid, 0.25), color=GRAY, lw=1.2, ls=(0, (5, 2.5)), zorder=2)
    ax.plot(q_grid, boundary, color=NAVY, lw=2.0, zorder=3)
    ax.plot(q_grid, critical_gate(q_grid, 1.0), color=GRAY, lw=1.2, ls=(0, (1.1, 2.2)), zorder=2)
    ax.axvline(Q0, color=GRAY, lw=0.65, ls=":", alpha=0.7, zorder=1)
    ax.scatter([Q0, Q0], [0.25, 0.50], c=[RED, TEAL], s=29, edgecolors="white", linewidths=0.7, zorder=5)
    ax.scatter([Q0], [a_critical], s=36, facecolors="white", edgecolors=NAVY, linewidths=1.4, zorder=6)
    ax.annotate(
        r"critical $a_0=0.36$",
        xy=(Q0, a_critical),
        xytext=(0.278, 0.423),
        fontsize=8,
        color=NAVY,
        arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 0.7},
        ha="left",
        va="center",
    )
    ax.text(0.225, 0.558, r"$a_0=0.50$", color=TEAL, ha="left", fontsize=8)
    ax.text(0.178, 0.158, r"$a_0=0.25$", color=RED, ha="right", fontsize=8)
    ax.text(0.37, 0.86, "Recovery", color=TEAL, ha="center", fontsize=9)
    ax.text(0.083, 0.075, "Starvation", color=RED, ha="center", fontsize=9)
    ax.set_title(r"(a) Exact boundary at $\lambda=1$", loc="left", fontweight="bold", pad=8)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel(r"Initial solver quality $q_0$")
    ax.set_ylabel(r"Initial attempt probability $a_0$")
    ax.legend(
        handles=[
            Line2D([0], [0], color=GRAY, lw=1.2, ls=(0, (5, 2.5)), label=r"$\kappa=0.25$"),
            Line2D([0], [0], color=NAVY, lw=2, label=r"$\kappa=0.5$"),
            Line2D([0], [0], color=GRAY, lw=1.2, ls=(0, (1.1, 2.2)), label=r"$\kappa=1$"),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.245),
        frameon=False,
        columnspacing=0.7,
        handlelength=1.8,
        handletextpad=0.5,
    )

    # Panel (b): two trajectories from the same conditional solver quality.
    ax = axes[1]
    for a0, color in ((0.25, RED), (0.50, TEAL)):
        time, a, q = curves[a0]
        ax.plot(time, a, color=color, lw=1.9)
        ax.plot(time, q, color=color, lw=1.65, ls=(0, (4, 2.2)))
    ax.axhline(QC, color=GRAY, lw=0.65, ls=":", alpha=0.85)
    ax.axhline(q_starved, color=RED, lw=0.6, ls=":", alpha=0.35)
    ax.text(39, 0.515, r"$q_c=0.5$", ha="right", va="bottom", color="#65717D", fontsize=8)
    ax.text(26, 0.345, r"$q_\infty=0.3085$", color=RED, fontsize=8.5)
    ax.text(27, 0.055, r"$a\to0$", color=RED, fontsize=8.5)
    ax.text(26, 0.885, r"$a,q\to1$", color=TEAL, fontsize=8.5)
    ax.set_title(r"(b) Dynamics: $\kappa=0.5,\ q_0=0.2$", loc="left", fontweight="bold", pad=8)
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.arange(0, HORIZON + 1, 10))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Model time (rescaled)")
    ax.set_ylabel("Probability")
    ax.legend(
        handles=[
            Line2D([0], [0], color=RED, lw=1.9, label=r"$a(t),\ a_0=0.25$"),
            Line2D([0], [0], color=RED, lw=1.65, ls=(0, (4, 2.2)), label=r"$q(t),\ a_0=0.25$"),
            Line2D([0], [0], color=TEAL, lw=1.9, label=r"$a(t),\ a_0=0.50$"),
            Line2D([0], [0], color=TEAL, lw=1.65, ls=(0, (4, 2.2)), label=r"$q(t),\ a_0=0.50$"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.245),
        frameon=False,
        columnspacing=0.85,
        handlelength=1.8,
        handletextpad=0.5,
        fontsize=7.8,
    )
    for ax in axes:
        ax.grid(color=GRID, alpha=0.55)
        ax.set_axisbelow(True)

    fig.text(
        0.285,
        0.025,
        r"Shading uses $\kappa=0.5$; equality is starvation.",
        ha="center",
        fontsize=7.4,
        color="#566371",
    )
    save(fig, "figA_penalty_barrier", dpi=240)


def method_overview(save):
    teal, blue, orange = "#128577", "#285f91", "#c9634c"
    ink, gray = "#243444", "#e9edf2"
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panels = [
        (
            0.015,
            "TRAIN",
            "Learn to solve",
            "Correctness reward\nTrain under token cap\nFreeze solver weights",
            blue,
        ),
        (
            0.355,
            "SCORE",
            "Use the answer span",
            "Returned token logprobs\nAverage largest 10%\nOne score per rollout",
            teal,
        ),
        (
            0.695,
            "CALIBRATE",
            "Set risk level",
            "Sample once per family\nRank wrong-answer scores\nSet cutoff " + r"$\widehat\tau$",
            teal,
        ),
    ]
    for x, header, title, body, color in panels:
        ax.text(
            x + 0.145, 0.965, header, ha="center", va="top", fontsize=12.5, color=color, weight="bold"
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.49),
                0.29,
                0.37,
                boxstyle="round,pad=.009,rounding_size=.012",
                facecolor="#f5f8fa",
                edgecolor=color,
                linewidth=1.4,
            )
        )
        ax.text(
            x + 0.145, 0.80, title, ha="center", va="center", fontsize=12.3, weight="bold", color=ink
        )
        ax.text(
            x + 0.145, 0.645, body, ha="center", va="center", fontsize=11.5, linespacing=1.55, color=ink
        )
    for start, end in [(0.309, 0.342), (0.649, 0.682)]:
        ax.add_patch(
            FancyArrowPatch(
                (start, 0.675),
                (end, 0.675),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.4,
                color=ink,
            )
        )
    ax.text(0.015, 0.375, "Single-rollout deployment", fontsize=13.5, weight="bold", color=ink)
    ax.text(
        0.985,
        0.375,
        r"Submit if $S<\widehat\tau$; otherwise abstain",
        ha="right",
        fontsize=11.5,
        color=ink,
    )
    ax.add_patch(Rectangle((0.02, 0.06), 0.26, 0.19, facecolor=gray, edgecolor="none"))
    ax.text(0.15, 0.155, "Thinking\n(unscored)", ha="center", va="center", fontsize=12.5, color=ink)
    ax.axvline(0.315, ymin=0.065, ymax=0.265, color=ink, linewidth=1.2)
    ax.text(0.315, 0.285, "</think>", ha="center", fontsize=11.5, fontfamily="monospace", color=ink)
    ax.add_patch(Rectangle((0.353, 0.06), 0.50, 0.19, facecolor="#e9f4f2", edgecolor="none"))
    heights = np.array(
        [
            0.20,
            0.29,
            0.19,
            0.36,
            0.25,
            0.42,
            0.88,
            0.24,
            0.31,
            0.19,
            0.39,
            0.28,
            0.26,
            0.94,
            0.24,
            0.35,
            0.21,
            0.30,
            0.23,
            0.18,
        ]
    )
    top = set(np.argsort(heights)[-2:])
    for i, height in enumerate(heights):
        ax.add_patch(
            Rectangle(
                (0.365 + i * 0.0237, 0.078),
                0.016,
                0.15 * height,
                facecolor=orange if i in top else teal,
                edgecolor="none",
            )
        )
    ax.text(
        0.615, 0.285, "Answer span: largest 2 of 20 surprises", ha="center", fontsize=12.5, color=ink
    )
    ax.axvline(0.887, ymin=0.065, ymax=0.265, color=ink, linewidth=1.2)
    ax.text(0.887, 0.285, "EOS", ha="center", fontsize=12, color=ink)
    ax.text(
        0.603,
        0.012,
        "Tail means surprise magnitude, not the last tokens",
        ha="center",
        va="bottom",
        fontsize=11.8,
        color=ink,
    )
    save(fig, "fig_method_overview", dpi=190, pad_inches=0.06)


def theory_intuition(case_rows, save):
    teal, orange, ink = "#128577", "#c9634c", "#243444"
    cases = [r for r in case_rows if r["case"] in (5, 10)]
    assert [r["case"] for r in cases] == [5, 10]
    assert all(
        (r["size"], r["budget"], r["observed"]["arm"], r["calibration_domain"], r["alpha"])
        == ("8B", 4096, "G0_41", "DAPO", 0.065)
        for r in cases
    )
    # Keep the cutoff and decision rule inside an explicit canvas footer.
    # This avoids relying on a negative axes coordinate and tight-crop padding.
    fig = plt.figure(figsize=(7.25, 3.45))
    ax = fig.add_axes([0.085, 0.225, 0.335, 0.625])
    q = np.linspace(0.025, 0.5, 500)
    critical_a = critical_gate(q, KAPPA)
    ax.fill_between(q, 0, critical_a, color="#f7e5e0")
    ax.fill_between(q, critical_a, 1, color="#e3f1ed")
    ax.plot(q, critical_a, color=ink, lw=1.8)
    ax.axvline(0.2, color="#7b8793", lw=0.8, ls="--")
    ax.scatter([0.2, 0.2], [0.25, 0.50], c=[orange, teal], s=44, zorder=5)
    ax.scatter([0.2], [0.36], c=[ink], marker="D", s=26, zorder=5)
    ax.annotate(
        r"$a_0=0.36$",
        xy=(0.2, 0.36),
        xytext=(0.275, 0.38),
        fontsize=9.5,
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": ink},
    )
    ax.text(0.34, 0.80, "Recovery", color=teal, fontsize=10, ha="center")
    ax.text(0.115, 0.13, "Starvation", color=orange, fontsize=9.5, ha="center")
    ax.set(
        xlim=(0.025, 0.5),
        ylim=(0, 1),
        xlabel=r"Initial solver quality $q_0$",
        ylabel=r"Initial attempt probability $a_0$",
    )
    ax.set_title("(a) Attempt starvation", fontsize=10, loc="left", pad=24, fontweight="bold", color=ink)
    ax.text(
        0, 1.035, r"Exact model: $\lambda=1,\ \kappa=0.5$", transform=ax.transAxes, fontsize=9, color=ink
    )
    ax.xaxis.label.set_fontsize(9.5)
    ax.yaxis.label.set_fontsize(9.5)
    ax.tick_params(labelsize=9)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(axis="y", color="#dce1e6", lw=0.5, alpha=0.6)
    ax = fig.add_axes([0.485, 0.225, 0.505, 0.625])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title(
        "(b) Two observed score decisions", fontsize=10, loc="left", pad=24, fontweight="bold", color=ink
    )
    ax.text(
        0, 1.035, "Qwen3-8B · Binary RL · seed 41 · 4K", fontsize=8.8, transform=ax.transAxes, color=ink
    )
    for case, bottom in zip(cases, [0.525, 0.045]):
        observed = case["observed"]
        correct = observed["vs"] == "R"
        candidate_color = teal if correct else orange
        ax.add_patch(
            FancyBboxPatch(
                (0.003, bottom),
                0.99,
                0.435,
                boxstyle="round,pad=.006,rounding_size=.018",
                facecolor="#f7fafb",
                edgecolor="#cdd6dc",
                linewidth=0.8,
            )
        )
        ax.add_patch(
            Rectangle((0.003, bottom + 0.013), 0.009, 0.409, facecolor=candidate_color, edgecolor="none")
        )
        domain = "Omni-MATH" if case["domain"] == "Omni" else "DAPO"
        ax.text(
            0.035,
            bottom + 0.367,
            f"Case {case['case']} · {domain}",
            fontsize=8.9,
            color=ink,
            weight="bold",
        )
        ax.text(
            0.96,
            bottom + 0.367,
            "CORRECT" if correct else "WRONG",
            fontsize=8.2,
            color=candidate_color,
            weight="bold",
            ha="right",
        )
        ax.text(
            0.035,
            bottom + 0.257,
            f'{observed["output_tokens"]:,} sampled output tokens',
            fontsize=8.7,
            color=ink,
            va="center",
        )
        for label, score_key, decision_key, y in [
            ("Box NLL", "box_sum_nll", "box_decision", bottom + 0.149),
            ("Tail NLL", "tail10_nll", "tail_decision", bottom + 0.049),
        ]:
            value = observed[score_key]
            if value >= 0.001:
                value_text = f"{value:.3f}"
            else:
                mantissa, exponent = f"{value:.3e}".split("e")
                value_text = rf"${mantissa}\times10^{{{int(exponent)}}}$"
            action = observed[decision_key]
            good_decision = (action == "accept") == correct
            color = teal if good_decision else orange
            ax.text(0.035, y, label, fontsize=9, color=ink, va="center")
            ax.text(0.48, y, value_text, fontsize=9, color=ink, va="center", ha="right")
            ax.text(
                0.58,
                y,
                action.capitalize() + (" correct" if correct else " wrong"),
                fontsize=8.8,
                color=color,
                weight="bold",
                va="center",
            )
    box_cutoff, tail_cutoff = (cases[0]["observed"][name + "_cutoff"] for name in ("box", "tail"))
    assert all(
        (r["observed"]["box_cutoff"], r["observed"]["tail_cutoff"]) == (box_cutoff, tail_cutoff)
        for r in cases
    )
    mantissa, exponent = f"{box_cutoff:.3e}".split("e")
    cutoff_text = (
        rf"DAPO cutoffs: Box ${mantissa}\times10^{{{int(exponent)}}}$; " rf"Tail ${tail_cutoff:.3f}$"
    )
    fig.text(0.485, 0.100, cutoff_text, fontsize=8.7, color=ink, va="bottom")
    fig.text(
        0.485, 0.040, r"Accept if score $<$ cutoff; $\alpha=0.065$", fontsize=8.7, color=ink, va="bottom"
    )
    save(fig, "fig_theory_intuition", bbox_inches=None, dpi=190)


def load_cases(path):
    """Validate the grade and strict score-cutoff decisions for the 12 selected cases."""
    rows = json.loads(path.read_text())
    assert [r["case"] for r in rows] == list(range(1, 13))
    for row in rows:
        observed = row["observed"]
        for rule, field in (("box", "box_sum_nll"), ("tail", "tail10_nll")):
            unbounded = observed[f"{rule}_cutoff_unbounded"]
            cutoff = observed[f"{rule}_cutoff"]
            assert (cutoff is None) == unbounded
            threshold = math.inf if unbounded else cutoff
            score = observed[field]
            if observed["vs"] == "U":
                assert score is None
                decision = "reject (invalid)"
            else:
                assert observed["vs"] in ("R", "W") and math.isfinite(score)
                decision = "accept" if score < threshold else "reject"
            assert observed[f"{rule}_decision"] == decision
    return rows
