#!/usr/bin/env python3
"""Figures for the Paper 1 manuscript -- reads results/*.csv, writes docs/figures/paper1/*.png.

Analysis output only; nothing in the pipeline is recomputed here except what a figure needs
to draw. Palette follows the dataviz method (validated: bird #eb6834 / drone #2a78d6 pass every
check on a light surface); single-series charts use ink, never a series hue; gridlines are
hairline and recessive; every multi-series chart carries a legend.

    python src/paper1/paper_figures.py                     # all figures
    python src/paper1/paper_figures.py --only 2,4,5        # the ones that need no sweep output

Figure 3 (solo-feature heatmap) and 6 (selection bias) read the feature-selection sweep, so
they must be drawn after scripts/run_tortuosity_fix_sweeps.sh has finished.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "paper1"

BIRD, DRONE = "#eb6834", "#2a78d6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#e8e7e1"
SURFACE = "#ffffff"                                       # print page
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FEATURES = ["r2_95", "r3_95", "r5_95", "speed_std", "accel_std", "angle_std", "tortuosity"]
FEAT_LABEL = {"r2_95": "R2", "r3_95": "R3", "r5_95": "R5", "speed_std": "speed σ",
              "accel_std": "accel. σ", "angle_std": "angle σ", "tortuosity": "tortuosity"}
TRACKERS = ["bytetrack", "occluboost", "botsort", "strongsort", "ocsort", "hybridsort",
            "boosttrack", "deepocsort", "sfsort"]
NAME = {"bytetrack": "ByteTrack", "occluboost": "OccluBoost", "botsort": "BoT-SORT",
        "strongsort": "StrongSORT", "ocsort": "OC-SORT", "hybridsort": "Hybrid-SORT",
        "boosttrack": "BoostTrack", "deepocsort": "Deep OC-SORT", "sfsort": "SFSORT",
        "ceiling": "Perfect tracks (GT)"}
WIDTH = 8.6 / 2.54           # inches = the 8.6 cm JCST column, so a figure goes in at 1:1 and its
                             # fonts print at the size set below (6.3 in = 16 cm until 2026-10-13)
RULES = ("vote", "rf")                                    # voting rule first, then the forest
RULE_TITLE = ("(a) Fitted voting rule", "(b) Random forest")
HILITE = "#4a3aa7"            # categorical violet: the one highlighted tracker (not a class hue)


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5, "axes.titlepad": 3,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.3,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  -> {OUT / name}")


# ------------------------------------------------------------ figure 1

def fig1_pipeline() -> None:
    """Pipeline schematic in three rows that fit one column, read as a snake: the tracker arm runs
    along the top and down the right, the perfect-track reference (dashed) takes the ground-truth
    boxes straight to the SAME feature code, and the classifier ends the bottom row on the left.
    Drawn in inches, so text sizes are the printed sizes."""
    H, gap, row_gap, bh = 1.62, 0.13, 0.17, 0.40
    bw = (WIDTH - 2 * gap) / 3
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH), ax.set_ylim(0, H), ax.axis("off")
    x = [i * (bw + gap) for i in range(3)]                    # left edge of each column
    y = [H - bh - k * (bh + row_gap) for k in range(3)]      # bottom edge of each row

    def box(x0, y0, w, title, sub, dashed=False):
        ax.add_patch(FancyBboxPatch((x0 + 0.01, y0), w - 0.02, bh,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=SURFACE if dashed else "#f4f3ef", ec=MUTED if dashed else AXIS,
                                    lw=0.7, ls=(0, (3, 2)) if dashed else "-"))
        ax.text(x0 + w / 2, y0 + bh * 0.66, title, ha="center", va="center", fontsize=7.5, color=INK)
        ax.text(x0 + w / 2, y0 + bh * 0.27, sub, ha="center", va="center", fontsize=6.5, color=INK2)

    def arrow(xy_from, xy_to, colour=INK2):
        ax.annotate("", xy=xy_to, xytext=xy_from,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.7, mutation_scale=6,
                                    shrinkA=0, shrinkB=0))

    box(x[0], y[0], bw, "Video clip", "Svanström, 10 s")
    box(x[1], y[0], bw, "YOLOv8s per fold", "1 class, conf ≥ 0.10")
    box(x[2], y[0], bw, "9 BoxMOT trackers", "start 0.45, floor 0.10")
    box(x[2], y[1], bw, "Track selection", "longest | oracle")
    box(x[0], y[1], 2 * bw + gap, "Ground-truth boxes", "perfect-track reference", dashed=True)
    box(x[1], y[2], bw, "7 trajectory features", "area, motion, path")
    box(x[0], y[2], bw, "Classifier", "fitted vote | RF")
    mid = lambda r: y[r] + bh / 2  # noqa: E731
    arrow((x[0] + bw - 0.01, mid(0)), (x[1] + 0.01, mid(0)))
    arrow((x[1] + bw - 0.01, mid(0)), (x[2] + 0.01, mid(0)))
    arrow((x[2] + bw / 2, y[0]), (x[2] + bw / 2, y[1] + bh))                      # trackers -> selection
    arrow((x[0] + bw / 2, y[0]), (x[0] + bw / 2, y[1] + bh), MUTED)               # clip -> GT boxes
    ax.plot([x[2] + bw / 2] * 2, [y[1], mid(2)], color=INK2, lw=0.7)              # selection -> features
    arrow((x[2] + bw / 2, mid(2)), (x[1] + bw - 0.01, mid(2)))
    arrow((x[1] + bw / 2, y[1]), (x[1] + bw / 2, y[2] + bh), MUTED)               # GT boxes -> features
    arrow((x[1] + 0.01, mid(2)), (x[0] + bw - 0.01, mid(2)))                      # features -> classifier
    save(fig, "fig1_pipeline.png")


# ------------------------------------------------------------ figure 2

def fig2_trackers() -> None:
    """Macro F1 with 95% CI per tracker on the common clip set, both decision rules
    (voting rule first, as in the manuscript; rows sorted by the random forest)."""
    ci = pd.read_csv(RES / "paper_stats_ci.csv")
    com = ci[(ci.set == "common") & (ci.policy == "longest")]
    order = (com[(com.rule == "rf") & (com.tracker != "ceiling")]
             .sort_values("macro_f1").tracker.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    for ax, rule, title in zip(axes, RULES, RULE_TITLE):
        sub = com[com.rule == rule].set_index("tracker")
        c = sub.loc["ceiling"]
        ax.axvspan(c.ci_lo, c.ci_hi, color=BAND, lw=0, zorder=0)
        ax.axvline(c.macro_f1, color=MUTED, lw=0.8, zorder=1)
        ax.text(c.macro_f1, len(order) - 0.3, "perfect tracks", color=INK2, fontsize=6.5,
                ha="center", va="bottom", bbox=dict(fc=SURFACE, ec="none", pad=0.6))
        for i, t in enumerate(order):
            r = sub.loc[t]
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=INK2, lw=1.0, solid_capstyle="round", zorder=2)
            ax.plot(r.macro_f1, i, "o", ms=3.8, color=INK, mec=SURFACE, mew=0.8, zorder=3)
        ax.axvline(1 / 3, color=AXIS, lw=0.7, ls=(0, (2, 2)), zorder=0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(0.3, 1.0)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Macro F1")                 # n and the CI are stated in the caption
        ax.grid(axis="x"), ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), [NAME[t] for t in order])
    axes[0].set_ylim(-0.6, len(order) + 0.25)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_tracker_macro_f1.png")


# ------------------------------------------------------------ figure 3

def fig3_solo_heatmap() -> None:
    """Each feature alone (macro F1): perfect tracks vs each tracker (longest), voting rule
    in panel (a) and random forest in panel (b)."""
    units = ["ceiling"] + [f"{t}_longest" for t in TRACKERS]
    solo = {u: pd.read_csv(RES / f"feature_selection_{u}.csv").query("k == 1").set_index("features")
            for u in units}
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.6))
    for ax, rule, title in zip(axes, RULES, RULE_TITLE):
        M = np.array([[solo[u].loc[f, f"{rule}_macro_f1"] for f in FEATURES] for u in units])
        im = ax.imshow(M, cmap=cmap, vmin=0.3, vmax=0.9, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if M[i, j] > 0.62 else INK)
        ax.set_xticks(range(len(FEATURES)), [FEAT_LABEL[f] for f in FEATURES],
                      rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(units)), [NAME[u.split("_")[0]] for u in units])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(0.5, color=SURFACE, lw=2.5)                 # separate the reference row
        ax.set_title(title, loc="left", color=INK)
    fig.tight_layout(h_pad=0.8)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, aspect=40)
    cb.set_label("Macro F1, feature alone")
    cb.outline.set_visible(False)
    save(fig, "fig3_solo_feature_heatmap.png")


# ------------------------------------------------------------ figure 4

def fig4_class_quality() -> None:
    """Tracked-box recall and coverage, bird vs drone, per tracker (medians; * = Holm p < 0.05)."""
    fr = pd.read_csv(RES / "paper_stats_fragmentation.csv").set_index("tracker")
    order = fr.sort_values("recall_bird_median").index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.15), sharey=True)
    for ax, key, title in zip(axes, ("recall", "coverage"),
                              ("(a) Tracked-box recall", "(b) Coverage")):
        for i, t in enumerate(order):
            b, d = fr.loc[t, f"{key}_bird_median"], fr.loc[t, f"{key}_drone_median"]
            ax.plot([b, d], [i, i], color=AXIS, lw=1.2, zorder=1)
            ax.plot(d, i, "o", ms=4.2, color=DRONE, mec=SURFACE, mew=0.8, zorder=3)
            ax.plot(b, i, "o", ms=4.2, color=BIRD, mec=SURFACE, mew=0.8, zorder=3)
            if fr.loc[t, f"{key}_p_holm"] < 0.05:
                ax.text(1.04, i, "*", va="center", ha="left", fontsize=8, color=INK)
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Median per clip")
        ax.grid(axis="x"), ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), [NAME[t] for t in order])
    axes[0].plot([], [], "o", ms=4.2, color=BIRD, label="Bird")
    axes[0].plot([], [], "o", ms=4.2, color=DRONE, label="Drone")
    axes[0].legend(loc="lower left", handletextpad=0.1)          # panel (a) is empty below 0.35
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_class_tracking_quality.png")


# ------------------------------------------------------------ figure 5

def fig5_tortuosity() -> None:
    """Tortuosity and net displacement per second by class: perfect tracks vs ByteTrack."""
    ceil = pd.read_csv(RES / "ceiling_features_track.csv")
    bt = pd.read_csv(RES / "tracker_features_bytetrack.csv")
    bt = bt[bt.policy == "longest"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 2.75), sharey="row")
    panels = [(ceil, "tortuosity", "(a) Tortuosity, GT"), (bt, "tortuosity", "(b) Tortuosity, ByteTrack"),
              (ceil, "net_disp_s", "(c) Net disp./s, GT"), (bt, "net_disp_s", "(d) Net disp./s, ByteTrack")]
    for ax, (df, col, title) in zip(axes.flat, panels):
        for k, (cls, color) in enumerate((("BIRD", BIRD), ("DRONE", DRONE))):
            v = df[df.cls == cls][col].values
            ax.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=color,
                       edgecolor=SURFACE, linewidth=0.3, zorder=2)
            ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color=INK, lw=1.3, zorder=3)
        ax.set_xticks([0, 1], ["Bird", "Drone"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y"), ax.set_axisbelow(True)
        if col == "net_disp_s":
            ax.set_yscale("log")
        else:
            ax.set_yticks([2, 4, 6, 8, 10])
    fig.tight_layout(w_pad=0.6, h_pad=0.6)                       # the caption says the bar is the median
    save(fig, "fig5_tortuosity_confound.png")


# ------------------------------------------------------------ figure 6

def fig6_selection_bias() -> None:
    """All seven features vs best-of-127 vs nested estimate; voting rule, then random forest."""
    ns = pd.read_csv(RES / "feature_selection_nested.csv")
    units = ["ceiling"] + [f"{t}_longest" for t in TRACKERS]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35), sharey=True, sharex=True)
    for ax, rule, title in zip(axes, RULES, RULE_TITLE):
        sub = ns[ns.rule == rule].set_index("unit")
        us = [u for u in units if u in sub.index]
        for i, u in enumerate(us):
            r = sub.loc[u]
            lo, hi = min(r.all7, r.nested, r.best_of_127), max(r.all7, r.nested, r.best_of_127)
            ax.plot([lo, hi], [i, i], color=AXIS, lw=1.0, zorder=1)
            ax.plot(r.best_of_127, i, marker="x", ms=4, color=MUTED, mew=1.0, ls="", zorder=2)
            ax.plot(r.all7, i, marker="o", ms=4, mfc=SURFACE, mec=INK2, mew=0.9, ls="", zorder=3)
            ax.plot(r.nested, i, marker="o", ms=4, color=INK, mec=SURFACE, mew=0.7, ls="", zorder=4)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Macro F1")
        ax.grid(axis="x"), ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(us)), [NAME[u.split("_")[0]] for u in us])
    axes[0].invert_yaxis()
    axes[0].margins(x=0.08)                                   # room for the outermost markers
    axes[0].xaxis.set_major_locator(MultipleLocator(0.1))
    axes[0].plot([], [], "x", ms=4, color=MUTED, mew=1.0, label="Best of 127 subsets (biased)")
    axes[0].plot([], [], "o", ms=4, mfc=SURFACE, mec=INK2, label="All seven features")
    axes[0].plot([], [], "o", ms=4, color=INK, label="Selection, nested (honest)")
    fig.tight_layout(w_pad=0.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=2, handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig6_selection_bias.png")


# ------------------------------------------------------------ figure 7

def fig7_roc() -> None:
    """ROC curves on the 92 common clips (longest policy): voting rule, then random forest.
    Perfect tracks in ink, ByteTrack highlighted in both panels (colour follows the tracker,
    not its rank), the other eight trackers as muted context. AUC in the legend."""
    cur = pd.read_csv(RES / "roc_curves.csv")
    auc = pd.read_csv(RES / "roc_auc.csv")
    cur = cur[(cur.policy == "longest") & (cur.set == "common")]
    auc = auc[(auc.policy == "longest") & (auc.set == "common")].set_index(["tracker", "rule"]).auc
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.05), sharey=True)
    for ax, rule, title in zip(axes, RULES, RULE_TITLE):
        c = cur[cur.rule == rule]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.7, ls=(0, (2, 2)), zorder=0)
        others = [t for t in TRACKERS if t != "bytetrack"]
        for t in others:
            g = c[c.tracker == t]
            ax.plot(g.fpr, g.tpr, color=MUTED, lw=0.6, alpha=0.7, zorder=1)
        lo, hi = min(auc[(t, rule)] for t in others), max(auc[(t, rule)] for t in others)
        g = c[c.tracker == "bytetrack"]
        ax.plot(g.fpr, g.tpr, color=HILITE, lw=1.2, zorder=3,
                label=f"ByteTrack {auc[('bytetrack', rule)]:.2f}")
        g = c[c.tracker == "ceiling"]
        ax.plot(g.fpr, g.tpr, color=INK, lw=1.5, zorder=4,
                label=f"Perfect tracks {auc[('ceiling', rule)]:.2f}")
        ax.plot([], [], color=MUTED, lw=0.6, label=f"Other eight {lo:.2f}–{hi:.2f}")
        h, lab = ax.get_legend_handles_labels()
        # white backing: in (a) the legend sits over the curves that hug the diagonal
        ax.legend([h[1], h[0], h[2]], [lab[1], lab[0], lab[2]], loc="lower right", title="AUC",
                  title_fontsize=6.5, fontsize=6.5, handlelength=1.2, labelspacing=0.25,
                  borderpad=0.25, alignment="right", frameon=True, facecolor=SURFACE,
                  edgecolor="none", framealpha=0.9)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
        ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("False-positive rate\n(drones called bird)", linespacing=1.1)
        ax.grid(True), ax.set_axisbelow(True)
    axes[0].set_ylabel("True-positive rate\n(birds called bird)", linespacing=1.1)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig7_roc.png")


FIGS = {1: fig1_pipeline, 2: fig2_trackers, 3: fig3_solo_heatmap, 4: fig4_class_quality,
        5: fig5_tortuosity, 6: fig6_selection_bias, 7: fig7_roc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated figure numbers")
    args = ap.parse_args()
    style()
    todo = [int(x) for x in args.only.split(",") if x] or sorted(FIGS)
    for n in todo:
        print(f"figure {n}")
        FIGS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
