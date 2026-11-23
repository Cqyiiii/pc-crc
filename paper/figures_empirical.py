"""Rebuild the five empirical paper figures from public replay measurements."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from replay.figure_data import read_csv

NAVY = "#275B8C"
TEAL = "#13867B"
RED = "#C75D4E"
GOLD = "#CC9C32"
GRAY = "#D8DDE3"
PURPLE = "#7B639D"


def select(rows, **filters):
    return [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]


def one(rows, **filters):
    selected = select(rows, **filters)
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected[0]


def three_state(points, penalties, save):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.05), sharex=True, sharey=True)
    labels = [
        "Native",
        "Budget forcing",
        "Binary RL",
        "Penalty RL",
        "Explicit-Abstain RL",
        "Tail-CRC (Ours)",
    ]
    state_labels = []
    for ax, dom in zip(axs, ["DAPO", "Omni"]):
        states = []
        for arm, decode in [("base", "Native"), ("base", "ExtBF"), ("G0_pooled", "ExtBF")]:
            row = one(
                points, domain=dom, size="8B", arm=arm, budget=4096, decode=decode, filter_rule="raw"
            )
            states.append([float(row[k]) for k in ("R_pct", "W_pct", "U_pct")])
        for arm in ["G1_41", "S3A"]:
            row = one(penalties, cell=f"{dom}/8B/BF/{arm}/4096")
            states.append([100 * float(row[k]) for k in ("R", "W", "U")])
        row = one(
            points,
            domain=dom,
            size="8B",
            arm="G0_pooled",
            budget=4096,
            decode="ExtBF",
            filter_rule="crc_tail10_0.065",
        )
        states.append([float(row[k]) for k in ("R_pct", "W_pct", "U_pct")])
        np.testing.assert_allclose(np.sum(states, axis=1), 100, rtol=0, atol=1.51e-6)
        y = np.arange(len(labels))
        states = np.array(states)
        left = np.zeros(len(labels))
        for j, color in enumerate([TEAL, RED, GRAY]):
            ax.barh(
                y, states[:, j], left=left, height=0.67, color=color, edgecolor="white", linewidth=0.4
            )
            if j < 2:
                for i, (l, w) in enumerate(zip(left, states[:, j])):
                    if w >= 5.5:
                        label = ax.text(
                            l + w / 2,
                            i,
                            f"{w:.1f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=8.5 if w < 12 else 9,
                        )
                        state_labels.append((ax, label, j, i, l, w, states[i, :2].sum()))
            left += states[:, j]
        ax.set_title(f"{dom}: Qwen3-8B, 4K", loc="left", fontweight="bold", pad=9)
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", length=0, pad=7)
        for label in ax.get_yticklabels():
            if label.get_text() == "Tail-CRC (Ours)":
                label.set_fontweight("bold")
                label.set_color(NAVY)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("All TEST families (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        # Keep net score outside the shared 0--100% outcome bars. Right alignment
        # gives every row the same decimal position; values come from the CSV.
        net_x = 1.20
        ax.text(
            net_x,
            1.045,
            "Net",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#59636E",
            clip_on=False,
        )
        for i, (right, wrong, _) in enumerate(states):
            ours = labels[i] == "Tail-CRC (Ours)"
            ax.text(
                net_x,
                i,
                f"{right - wrong:+.1f}".replace("-", "−"),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold" if ours else "normal",
                color=NAVY if ours else "#37414D",
                clip_on=False,
            )
    axs[0].invert_yaxis()
    fig.legend(
        handles=[
            Patch(color=TEAL, label="Right (R)"),
            Patch(color=RED, label="Wrong (W)"),
            Patch(color=GRAY, label="Unsubmitted / rejected (U)"),
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.5,
    )
    fig.tight_layout(w_pad=2.0, rect=(0, 0.095, 1, 1))
    # Check labels at the final axes size. When a segment cannot provide a 2 pt
    # inset on each side, keep the readable font size and put its R/W label in
    # the gray remainder, 2.5 pt past the combined right/wrong boundary.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inset_px = 2.0 * fig.dpi / 72
    callout_pad_px = 2.5 * fig.dpi / 72
    for ax, label, state, row, left, width, submitted in state_labels:
        x0, x1 = ax.transData.transform([(left, row), (left + width, row)])[:, 0]
        if label.get_window_extent(renderer).width + 2 * inset_px > x1 - x0:
            boundary_px, y_px = ax.transData.transform((submitted, row))
            label.set_position(ax.transData.inverted().transform((boundary_px + callout_pad_px, y_px)))
            label.set_text(f"{'R' if state == 0 else 'W'} {width:.1f}")
            label.set_ha("left")
            label.set_color(TEAL if state == 0 else RED)
    save(fig, "fig1_three_state")


def risk_coverage(front, penalties, save):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.65), sharex=True, sharey=True)
    styles = {
        "G1_41": (PURPLE, "D", "Penalty RL"),
        "S3A": (RED, "s", "Explicit-Abstain RL"),
        "S3B": (GOLD, "^", "Mild-Penalty RL"),
    }
    for ax, (dom, bud) in zip(
        axs.flat, [("DAPO", 4096), ("DAPO", 8192), ("Omni", 4096), ("Omni", 8192)]
    ):
        sweep = sorted(
            (r for r in front if r["domain"] == dom and r["budget"] == bud), key=lambda r: r["k"]
        )
        assert [r["k"] for r in sweep] == list(range(len(sweep))) and sweep
        ax.plot(
            [100 * r["coverage"] for r in sweep],
            [
                100 * r["conditional_error"] if r["conditional_error"] is not None else np.nan
                for r in sweep
            ],
            color=NAVY,
            lw=1.8,
        )
        for arm, (color, m, label) in styles.items():
            r = one(penalties, cell=f"{dom}/8B/BF/{arm}/{bud}")
            ax.scatter(
                100 * float(r["coverage"]),
                100 * float(r["conditional_error"]),
                s=42,
                marker=m,
                c=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )
        ax.set_title(f"{dom}, {bud//1024}K", loc="left", fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 65)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks(np.arange(0, 61, 20))
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
    for ax in axs[-1]:
        ax.set_xlabel("Submitted coverage (%)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Error among\nsubmissions (%)")
    fig.legend(
        handles=[Line2D([0], [0], color=NAVY, lw=1.8, label="Tail-CRC (Ours), threshold sweep")]
        + [
            Line2D([0], [0], marker=m, color="none", markerfacecolor=c, markersize=6, label=l)
            for c, m, l in styles.values()
        ],
        ncol=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.53, -0.035),
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.6, w_pad=1)
    save(fig, "fig2_risk_coverage")


def score_anatomy(uncertainty, save):
    assert len(uncertainty) == 36
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.12), gridspec_kw={"width_ratios": [1.15, 1.1, 1.25]})
    a = one(uncertainty, domain="DAPO", size="4B", arm="base", budget=4096)
    vals = [
        float(a[k]) for k in ("ans_sum_nll_auc", "ans_len_auc", "ans_mean_nll_auc", "tail10_nll_auc")
    ]
    axs[0].barh(np.arange(4), vals, color=[RED, GRAY, NAVY, TEAL], height=0.65)
    axs[0].set_yticks(np.arange(4), ["Sum NLL", "Length", "Mean NLL", "Tail NLL"])
    for i, v in enumerate(vals):
        axs[0].text(v + 0.025, i, f"{v:.3f}", va="center", fontsize=9)
    axs[0].axvline(0.5, color="black", ls=":", lw=0.8)
    axs[0].set_xlim(0, 1)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Wrong-answer AUROC")
    axs[0].set_title(
        "(a) Length effect\n4B Budget forcing\nDAPO 4K", loc="left", fontweight="bold", fontsize=9.3
    )

    for bud, marker, color in [(4096, "o", NAVY), (8192, "s", TEAL)]:
        for arm in ["G0_41", "G0_73"]:
            r = one(uncertainty, domain="Omni", size="8B", arm=arm, budget=bud)
            values = [float(r[k]) for k in ("box_sum_nll_auc", "ans_mean_nll_auc", "tail10_nll_auc")]
            axs[1].plot([0, 1, 2], values, color=color, marker=marker, lw=1, alpha=0.8, ms=4)
    axs[1].set_xticks([0, 1, 2], ["Box\nNLL", "Mean\nNLL", "Tail\nNLL"])
    axs[1].set_ylim(0.48, 0.86)
    axs[1].axhline(0.5, color="black", ls=":", lw=0.8)
    axs[1].set_ylabel("Wrong-answer AUROC")
    axs[1].set_title(
        "(b) Omni ranking\n8B Binary RL\n4K and 8K", loc="left", fontweight="bold", fontsize=9.3
    )
    axs[1].legend(
        handles=[
            Line2D([0], [0], color=NAVY, marker="o", label="4K"),
            Line2D([0], [0], color=TEAL, marker="s", label="8K"),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
        handlelength=0.8,
        columnspacing=0.6,
    )

    groups = [("4B", "DAPO"), ("8B", "DAPO"), ("4B", "Omni"), ("8B", "Omni")]
    for i, (sz, dom) in enumerate(groups):
        rows = sorted(
            select(uncertainty, size=sz, domain=dom), key=lambda r: (r["arm"], int(r["budget"]))
        )
        ys = i + np.linspace(-0.25, 0.25, len(rows))
        for yy, r in zip(ys, rows):
            lower, upper = float(r["diff_lower"]), float(r["diff_upper"])
            color = TEAL if lower > 0 else "#8996A4"
            axs[2].plot([lower, upper], [yy, yy], color=color, lw=0.75, alpha=0.85)
            axs[2].scatter(float(r["H_minus_receipt"]), yy, s=11, color=color, zorder=3)
    axs[2].axvline(0, color="black", ls=":", lw=0.8)
    axs[2].set_yticks(range(4), ["4B DAPO", "8B DAPO", "4B Omni", "8B Omni"])
    axs[2].invert_yaxis()
    axs[2].set_xlabel("Tail entropy $-$ Tail NLL\n(AUROC difference)", fontsize=9.5)
    axs[2].set_title(
        "(c) Entropy vs. NLL\n36 paired intervals\nDAPO and Omni",
        loc="left",
        fontweight="bold",
        fontsize=9.3,
    )
    axs[2].set_xlim(
        min(-0.16, min(float(r["diff_lower"]) for r in uncertainty) - 0.01),
        max(0.16, max(float(r["diff_upper"]) for r in uncertainty) + 0.01),
    )
    axs[2].set_xticks([-0.1, 0, 0.1, 0.2])
    for ax in axs:
        ax.grid(axis="x" if ax != axs[1] else "y", alpha=0.17)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.7)
    save(fig, "fig3_score_anatomy")


def calibration(points, transfer, ltt, ncal, save):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    for b, offset, color in [(4096, -0.17, NAVY), (8192, 0.17, TEAL)]:
        raw = one(
            points,
            domain="Omni",
            size="8B",
            arm="G0_pooled",
            budget=b,
            decode="ExtBF",
            filter_rule="raw",
        )
        vv = [float(raw["W_pct"])]
        for track in ["A", "B"]:
            r = one(
                transfer,
                size="8B",
                budget=b,
                arm="G0_pooled",
                track=track,
                score="tail10_nll",
                alpha=0.065,
            )
            vv.append(100 * float(r["W"]))
        axs[0].bar(np.arange(3) + offset, vv, width=0.31, color=color, label=f"{b//1024}K")
        for x, v in zip(np.arange(3) + offset, vv):
            label_x = x + (-0.035 if b == 4096 else 0.035)
            axs[0].text(label_x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5)
    axs[0].axhline(6.5, color=RED, ls="--", lw=0.8)
    axs[0].set_xticks([0, 1, 2], ["Raw", "Source", "Target"])
    axs[0].set_ylim(0, 48)
    axs[0].set_ylabel("Wrong / all TEST families (%)")
    axs[0].set_title("(a) Calibration source", loc="left", fontweight="bold", fontsize=9.3)
    axs[0].legend(
        frameon=False, fontsize=8.5, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8
    )

    for dom, offset, color in [("DAPO", -0.17, NAVY), ("Omni", 0.17, TEAL)]:
        issued = []
        for target in (0.1, 0.15, 0.2, 0.3):
            rows = select(ltt, domain=dom, score="tail10_nll", target=target)
            assert len(rows) == 18 and {r["issued"] for r in rows} <= {"True", "False"}
            issued.append(sum(r["issued"] == "True" for r in rows))
        axs[1].bar(np.arange(4) + offset, issued, width=0.31, color=color, label=dom)
        for x, v in zip(np.arange(4) + offset, issued):
            axs[1].text(x, v + 0.5, str(v), ha="center", fontsize=8.5)
    axs[1].set_xticks(np.arange(4), ["10", "15", "20", "30"])
    axs[1].set_xlabel("Conditional-error target (%)", fontsize=9.5)
    axs[1].set_ylabel("Issued / 18 configurations")
    axs[1].set_ylim(0, 21)
    axs[1].set_title("(b) LTT issuance", loc="left", fontweight="bold", fontsize=10)
    axs[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    for dom, color in [("DAPO", NAVY), ("Omni", TEAL)]:
        rows = sorted(
            select(ncal, domain=dom, size="8B", budget=4096, alpha=0.065), key=lambda r: int(r["ncal"])
        )
        assert len(rows) == (5 if dom == "DAPO" else 4)
        axs[2].plot(
            [int(r["ncal"]) for r in rows],
            [100 * float(r["sd"]) for r in rows],
            marker="o",
            ms=3.5,
            color=color,
            label=dom,
        )
    axs[2].set_xlabel("Calibration families")
    axs[2].set_ylabel("Test wrong-rate SD (pp)")
    axs[2].set_title("(c) Calibration size", loc="left", fontweight="bold", fontsize=10)
    axs[2].legend(frameon=False, fontsize=8.5)
    for ax in axs:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig4_calibration")


def training(steps, active_groups, save):
    """Trailing means on the three training arms shown in Figure 8."""
    arms = [
        ("8B_G0_41", "Binary RL (seed 41)", NAVY),
        ("8B_G0_73", "Binary RL (seed 73)", "#7DA3C2"),
        ("8B_G1_41", "Penalty RL (seed 41)", PURPLE),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(7, 4.6), sharex=True)
    for arm, label, color in arms:
        rows = sorted(select(steps, arm=arm), key=lambda r: int(r["step"]))
        active = sorted(select(active_groups, arm=arm), key=lambda r: int(r["step"]))
        step = [int(r["step"]) for r in rows]
        assert step == [int(r["step"]) for r in active] == list(range(1, 51))
        right = np.array([float(r["R"]) for r in rows])
        wrong = np.array([float(r["W"]) for r in rows])
        series = [
            (right, "Right fraction"),
            (1 - right - wrong, "Unsubmitted fraction"),
            ([float(r["active_group_fraction"]) for r in active], "Active reward groups"),
            ([float(r["mean_generated"]) for r in rows], "Generated tokens"),
        ]
        for ax, (values, ylabel) in zip(axs.flat, series):
            smooth = [np.mean(values[max(0, i - 4) : i + 1]) for i in range(len(step))]
            ax.plot(step, smooth, label=label, color=color, lw=1.2)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    for ax in axs[-1]:
        ax.set_xlabel("Training step")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.015),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "figA_training_curves")


def render(data, inputs, save):
    points = read_csv(data / "table1_dapo_omni_main_results.csv")
    penalties = read_csv(data / "table3_penalty_ablation_8b.csv")
    three_state(points, penalties, save)
    risk_coverage(json.loads((data / "frontier.json").read_text()), penalties, save)
    score_anatomy(read_csv(data / "table4_uncertainty_auroc_and_whitebox.csv"), save)
    calibration(
        points,
        read_csv(data / "table5_trackA_vs_trackB_and_ltt.csv"),
        read_csv(data / "ltt_all.csv"),
        read_csv(data / "ncal_all.csv"),
        save,
    )
    training(
        read_csv(data / "training_steps.csv"), read_csv(inputs / "training_active_groups.csv"), save
    )
