"""
visualization of the four hash-chain attack scenarios from
hash_chain_demo.py. produces a single multi-panel figure where
each panel shows the chain as a sequence of boxes with the
violated link highlighted.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


STAGE_NAMES = [
    "circuit_def",
    "transpile",
    "backend_sel",
    "calibration",
    "execution",
    "meas_output",
]

OK_FILL = "#cfe2d8"
OK_EDGE = "#3b8a5e"
BAD_FILL = "#f2cdc9"
BAD_EDGE = "#b03030"
INJ_FILL = "#fff2b0"
INJ_EDGE = "#b08020"
NEUTRAL_FILL = "#e7eaf0"
NEUTRAL_EDGE = "#5c6374"


def draw_chain(ax, labels, status, title, broken_link_after=None):
    """draw a chain of stages on `ax`.

    `status` is a list parallel to `labels` with values:
      'ok', 'tampered', 'injected', 'missing'
    `broken_link_after` is the index after which the arrow should
    be drawn red (link inconsistency).
    """
    n = len(labels)
    box_w = 1.5
    box_h = 0.9
    gap = 0.7
    total_w = n * box_w + (n - 1) * gap

    ax.set_xlim(-0.5, total_w + 0.5)
    ax.set_ylim(-0.5, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=10, loc="left", fontweight="bold")

    centers = []
    for i, (lbl, st) in enumerate(zip(labels, status)):
        x = i * (box_w + gap)
        if st == "ok":
            fc, ec = OK_FILL, OK_EDGE
        elif st == "tampered":
            fc, ec = BAD_FILL, BAD_EDGE
        elif st == "injected":
            fc, ec = INJ_FILL, INJ_EDGE
        elif st == "missing":
            fc, ec = NEUTRAL_FILL, NEUTRAL_EDGE
        else:
            fc, ec = NEUTRAL_FILL, NEUTRAL_EDGE

        if st == "missing":
            box = FancyBboxPatch(
                (x, 0), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.2, linestyle="--",
                edgecolor=ec, facecolor="white",
            )
        else:
            box = FancyBboxPatch(
                (x, 0), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.4,
                edgecolor=ec, facecolor=fc,
            )
        ax.add_patch(box)

        ax.text(
            x + box_w / 2, box_h / 2, lbl,
            ha="center", va="center",
            fontsize=8.5,
            fontstyle="italic" if st == "missing" else "normal",
            color="#666666" if st == "missing" else "black",
        )
        if st == "tampered":
            ax.text(x + box_w / 2, box_h + 0.18, "TAMPERED",
                    ha="center", fontsize=8, color=BAD_EDGE,
                    fontweight="bold")
        elif st == "injected":
            ax.text(x + box_w / 2, box_h + 0.18, "INJECTED",
                    ha="center", fontsize=8, color=INJ_EDGE,
                    fontweight="bold")
        elif st == "missing":
            ax.text(x + box_w / 2, box_h + 0.18, "SKIPPED",
                    ha="center", fontsize=8, color=NEUTRAL_EDGE,
                    fontweight="bold", fontstyle="italic")
        centers.append((x + box_w / 2, box_h / 2))

    for i in range(n - 1):
        x_start = centers[i][0] + box_w / 2
        x_end = centers[i + 1][0] - box_w / 2
        y = box_h / 2
        is_broken = (broken_link_after is not None
                     and i == broken_link_after)
        color = BAD_EDGE if is_broken else "#666666"
        lw = 2.0 if is_broken else 1.0
        ax.annotate(
            "",
            xy=(x_end, y), xytext=(x_start, y),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=lw,
                shrinkA=2, shrinkB=2,
            ),
        )
        if is_broken:
            mid_x = (x_start + x_end) / 2
            ax.text(mid_x, y + 0.18, "X", ha="center",
                    fontsize=14, color=BAD_EDGE, fontweight="bold")
            ax.text(mid_x, y - 0.28, "broken link",
                    ha="center", fontsize=8, color=BAD_EDGE,
                    fontstyle="italic")


def make_figure():
    fig, axes = plt.subplots(4, 1, figsize=(11, 9))

    draw_chain(
        axes[0],
        STAGE_NAMES,
        ["ok"] * 6,
        "Scenario 0  --  honest pipeline: every recomputed hash "
        "matches the stored hash; verifier returns OK",
    )

    draw_chain(
        axes[1],
        STAGE_NAMES,
        ["ok", "ok", "ok", "tampered", "ok", "ok"],
        "Scenario 1  --  tampering: spec at index 3 modified "
        "after the fact; recomputed hash fails to match",
    )

    inj_labels = (
        STAGE_NAMES[:4]
        + ["post_cal_patch"]
        + STAGE_NAMES[4:]
    )
    inj_status = ["ok"] * 4 + ["injected"] + ["ok"] * 2
    draw_chain(
        axes[2],
        inj_labels,
        inj_status,
        "Scenario 2  --  injection: a fake stage is spliced in; "
        "next legitimate record's prev_hash linkage breaks",
        broken_link_after=4,
    )

    skip_labels = list(STAGE_NAMES)
    skip_labels[3] = "(missing)"
    skip_status = ["ok", "ok", "ok", "missing", "ok", "ok"]
    draw_chain(
        axes[3],
        skip_labels,
        skip_status,
        "Scenario 3  --  skipping: a legitimate stage record is "
        "removed; downstream prev_hash no longer matches",
        broken_link_after=2,
    )

    legend_handles = [
        mpatches.Patch(facecolor=OK_FILL, edgecolor=OK_EDGE,
                       label="honest stage"),
        mpatches.Patch(facecolor=BAD_FILL, edgecolor=BAD_EDGE,
                       label="tampered stage"),
        mpatches.Patch(facecolor=INJ_FILL, edgecolor=INJ_EDGE,
                       label="injected stage"),
        mpatches.Patch(facecolor="white", edgecolor=NEUTRAL_EDGE,
                       label="skipped (missing) stage", linestyle="--"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=9,
    )

    fig.suptitle(
        "Hash-chain integrity verification on a hybrid QPU pipeline",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


if __name__ == "__main__":
    fig = make_figure()
    out = "hash_chain_scenarios.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Figure saved to {out}")
