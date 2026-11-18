#!/usr/bin/env python3
"""Image figures for the Paper 1 manuscript: what the data, the tracks and the failures look like.

Svanström frames only (the dataset is CC0, svanstrom_dataset/LICENSE). Drone-vs-Bird frames
are never drawn: they fall under the organisers' data usage agreement.

Frames are read from datasets/svan_det/images/<video>/<fno:06d>.jpg -- the exact images the
detector and every tracker saw, numbered 1-based like gt.txt -- so no video decoding and
no seek error can shift a box off its target.

No clip or frame is picked by eye. Each figure chooses by a stated rule and prints the
numbers that chose it, so the caption can quote the rule:
  dataset   per class x distance bin, the clip whose median GT box area is the bin median
            (lower middle for an even count); the middle frame among those whose label a
            cached detection confirms (IoU >= 0.3) -- the first version took the middle
            labelled frame and landed on a drifted label (V_BIRD_019 f188, ADR-006)
  strip     the bird clip with the largest StrongSORT-minus-ByteTrack count of tracks on
            the bird (results/tracker_quality.csv); four GT frames at fixed quantiles
  failure   (a) of the bird clips some tracker never follows (oracle policy,
                results/tracker_dropped_clips.csv), the one with the fewest detections >= 0.45,
                at its median-confidence detection on the bird (was V_BIRD_016 until v2)
            (b) the usable bird clip with the lowest detector recall (IoU >= 0.5): the
                middle frame with no detection within 1.5 box diagonals of the label
            (c) a source label removed by the ADR-006 audit: V_BIRD_020 (never on a bird)
                from gt/_orig_2026-09-24, at the frame where the most detections lie
                outside the label -- birds beside an empty box
            (d) the same clip's longest ByteTrack track, which follows another bird

Track-ID colours take the categorical order from slot 3 on (aqua, yellow, magenta, green,
violet, red): slots 1-2 are blue and orange, which this paper reserves for drone and bird.
IDs past the sixth fold to grey. Every track box also carries its ID as text, so identity
is never colour alone.

    python src/paper1/qualitative_figures.py                  # all three
    python src/paper1/qualitative_figures.py --only strip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracker_features as tf  # noqa: E402
from paper_figures import BIRD, DRONE, INK, MUTED, WIDTH, save, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "datasets" / "svan_det" / "images"
GT_DIR = ROOT / "gt" / "svanstrom_visible"
GT_ORIG = ROOT / "gt" / "_orig_2026-09-24" / "svanstrom_visible"     # before the ADR-006 audit
RES = ROOT / "results"
MIN_FRAMES = 6                                                       # feature_ceiling.MIN_FRAMES

ID_COLOURS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_ID = "#898781"
DET = "#eda100"                    # a raw detection (failure figure only; no tracks drawn there)
GT_EDGE = "#ffffff"
CONF_START = 0.45                  # run_trackers.CONF_HIGH
BINS = ["CLOSE", "MEDIUM", "DISTANT"]
HALO = [patheffects.withStroke(linewidth=1.6, foreground="#000000")]
TITLE = dict(fontsize=7, loc="left", color=INK, pad=2, linespacing=1.15)   # panel titles, printed size


# ------------------------------------------------------------------ data access

def manifest() -> pd.DataFrame:
    man = pd.read_csv(GT_DIR / "_manifest.csv")
    return man[(man.selected == 1) & man["class"].isin(["BIRD", "DRONE"])].reset_index(drop=True)


def frame(video: str, fno: int) -> np.ndarray:
    p = IMAGES / video / f"{fno:06d}.jpg"
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gt_boxes(video: str, root: Path = GT_DIR) -> dict[int, tuple]:
    """{frame: (x, y, w, h)} -- Svanström clips are single-target."""
    return {f: b[0][1:] for f, b in tf.read_mot(root / video / "gt" / "gt.txt").items()}


def confirmed_frames(video: str, fold: int, thr: float = 0.3) -> list[int]:
    """Labelled frames whose box a cached detection overlaps at IoU >= thr."""
    g, D = gt_boxes(video), dets(video, fold)
    return [f for f in sorted(g) if any(tf.iou(g[f], xywh(d)) >= thr for d in D[D[:, 0] == f])]


def dets(video: str, fold: int) -> np.ndarray:
    """Cached detections (frame, x1, y1, x2, y2, conf), as every tracker received them."""
    return np.load(RES / "dets" / f"fold{fold}" / f"{video}.npy")


def xywh(d) -> tuple:
    return float(d[1]), float(d[2]), float(d[3] - d[1]), float(d[4] - d[2])


def detector_recall(video: str, fold: int) -> float:
    g, D = gt_boxes(video), dets(video, fold)
    hit = sum(1 for f, b in g.items()
              if any(tf.iou(b, xywh(d)) >= 0.5 for d in D[D[:, 0] == f]))
    return hit / len(g)


# ------------------------------------------------------------------ drawing

def draw_box(ax, box, colour, dashed=False, label=None, lw=1.1, fs=6.5, label_below=False):
    x, y, w, h = box
    ls = (0, (2.5, 1.5)) if dashed else "-"
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#000000", lw=lw + 1.3, ls=ls))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=colour, lw=lw, ls=ls))
    if label:
        ty, va = (y + h + 1.5, "top") if label_below else (y - 1.5, "bottom")
        ax.text(x, ty, label, color=colour, fontsize=fs, va=va, ha="left",
                path_effects=HALO, clip_on=True)


def crop_window(box, half: float, W: int, H: int) -> tuple[int, int, int, int]:
    """Square window centred on the box, shifted (not shrunk) to stay inside the frame."""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    half = min(half, W / 2, H / 2)
    x0 = int(round(min(max(cx - half, 0), W - 2 * half)))
    y0 = int(round(min(max(cy - half, 0), H - 2 * half)))
    return x0, y0, x0 + int(2 * half), y0 + int(2 * half)


def show_crop(ax, img, win) -> None:
    x0, y0, x1, y1 = win
    ax.imshow(img[y0:y1, x0:x1], interpolation="nearest", extent=(x0, x1, y1, y0))
    ax.set_xlim(x0, x1), ax.set_ylim(y1, y0)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(MUTED), s.set_linewidth(0.6)


def full_with_zoom(ax, img, target, boxes, half=40.0) -> None:
    """Full frame with its boxes, plus a nearest-neighbour zoom on `target` in the corner
    diagonally opposite it (so the inset never covers the target). `boxes` is a list of
    (box, draw_box kwargs), drawn in both the frame and the zoom."""
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.set_xlim(0, W), ax.set_ylim(H, 0)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for b, kw in boxes:
        draw_box(ax, b, **{**kw, "label": None})
    win = crop_window(target, half, W, H)
    ax.add_patch(Rectangle((win[0], win[1]), win[2] - win[0], win[3] - win[1], fill=False,
                           ec="#ffffff", lw=0.6, ls=(0, (1, 1))))
    cx, cy = target[0] + target[2] / 2, target[1] + target[3] / 2
    ih = 0.42 * W / H * (win[3] - win[1]) / (win[2] - win[0])      # inset height, axes fraction
    ix = 0.02 if cx > W / 2 else 0.56
    iy = 0.02 if cy < H / 2 else 0.98 - ih      # axes y runs upwards: 0.02 = bottom; never past the top
    ins = ax.inset_axes([ix, iy, 0.42, ih])
    show_crop(ins, img, win)
    for s in ins.spines.values():
        s.set_color("#ffffff"), s.set_linewidth(1.0)
    for b, kw in boxes:
        draw_box(ins, b, **{**kw, "lw": 1.0})
    ins.text(0.03, 0.03, f"×{W * 0.42 / (win[2] - win[0]):.1f}", transform=ins.transAxes,
             color="#ffffff", fontsize=6.5, va="bottom", path_effects=HALO)


# ------------------------------------------------------------------ figure A: dataset

def pick_dataset_examples(man: pd.DataFrame) -> dict:
    rows = []
    for _, r in man.iterrows():
        g = gt_boxes(r.video)
        if len(g) < MIN_FRAMES:                      # no usable GT after the audit (020, 021)
            continue
        rows.append({"video": r.video, "cls": r["class"], "bin": r.distance_bin, "fold": int(r.fold),
                     "med_area": float(np.median([b[2] * b[3] for b in g.values()]))})
    d = pd.DataFrame(rows)
    picks = {}
    for (c, b), grp in d.groupby(["cls", "bin"]):
        grp = grp.sort_values(["med_area", "video"]).reset_index(drop=True)
        p = grp.iloc[(len(grp) - 1) // 2]
        fs = confirmed_frames(p.video, int(p.fold)) or sorted(gt_boxes(p.video))
        picks[(c, b)] = (p.video, fs[len(fs) // 2])
        print(f"  dataset {c:5s} {b:7s} n={len(grp):2d} bin-median side "
              f"{np.sqrt(p.med_area):5.1f} px -> {p.video} frame {fs[len(fs) // 2]} "
              f"({len(fs)} confirmed frames)")
    return picks


def fig_dataset_examples() -> None:
    picks = pick_dataset_examples(manifest())
    # one row per distance bin, bird left and drone right: two frames across a column keep the
    # frames large enough to see (three across left each one under 3 cm wide)
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 3 * (WIDTH / 2 * 512 / 640 + 0.3)))
    for i, (cls, colour) in enumerate((("BIRD", BIRD), ("DRONE", DRONE))):
        for j, b in enumerate(BINS):
            video, fno = picks[(cls, b)]
            box = gt_boxes(video)[fno]
            ax = axes[j, i]
            full_with_zoom(ax, frame(video, fno), box, [(box, {"colour": colour, "lw": 1.0})])
            ax.set_title(f"{cls.title()}, {b.lower()}\n{video}, frame {fno} · {box[2]:.0f}×{box[3]:.0f} px",
                         **TITLE)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.005, wspace=0.03, hspace=0.2)
    save(fig, "fig_dataset_examples.png")


# ------------------------------------------------------------------ figure B: tracking strip

def pick_strip_clip() -> str:
    q = pd.read_csv(RES / "tracker_quality.csv")
    p = q[q.cls == "BIRD"].pivot_table(index="video", columns="tracker", values="n_tracks_on_gt")
    diff = (p.strongsort - p.bytetrack).sort_values(ascending=False)
    top = diff.index[0]
    print(f"  strip: {top} StrongSORT {p.loc[top, 'strongsort']:.0f} vs ByteTrack "
          f"{p.loc[top, 'bytetrack']:.0f} tracks on the bird (next: {diff.index[1]}, "
          f"diff {diff.iloc[1]:.0f})")
    return top


def fig_tracking_strip() -> None:
    video = pick_strip_clip()
    fold = int(manifest().set_index("video").loc[video, "fold"])
    g = gt_boxes(video)
    fs = sorted(g)
    frames = [fs[int(round(qt * (len(fs) - 1)))] for qt in (0.15, 0.38, 0.62, 0.85)]
    D = dets(video, fold)
    for f in frames:                                        # frame-alignment check
        d = D[D[:, 0] == f]
        best = max((tf.iou(g[f], xywh(x)) for x in d), default=0.0)
        print(f"    frame {f}: IoU(GT, best detection) = {best:.2f}")
    q = pd.read_csv(RES / "tracker_quality.csv").set_index(["tracker", "video"])
    trackers = ("bytetrack", "strongsort")
    names = {"bytetrack": "ByteTrack", "strongsort": "StrongSORT"}
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, WIDTH * 0.9 / 4 * 2 + 0.3))
    for i, trk in enumerate(trackers):
        tk = tf.read_mot(RES / "tracks" / trk / f"{video}.txt")
        colour_of: dict[int, str] = {}
        for j, f in enumerate(frames):
            img = frame(video, f)
            win = crop_window(g[f], 44.0, img.shape[1], img.shape[0])
            ax = axes[i, j]
            show_crop(ax, img, win)
            draw_box(ax, g[f], GT_EDGE, dashed=True, lw=0.9)
            for tid, x, y, w, h in tk.get(f, []):
                if x + w < win[0] or x > win[2] or y + h < win[1] or y > win[3]:
                    continue
                if tid not in colour_of:
                    n = len(colour_of)
                    colour_of[tid] = ID_COLOURS[n] if n < len(ID_COLOURS) else OTHER_ID
                draw_box(ax, (x, y, w, h), colour_of[tid], label=f"#{tid}", lw=1.0, fs=6.5,
                         label_below=y - win[1] < 12)
            if i == 0:
                ax.set_title(f"frame {f}", fontsize=7, color=INK, pad=2)
        n_on = int(q.loc[(trk, video), "n_tracks_on_gt"])
        axes[i, 0].set_ylabel(f"{names[trk]}\n{n_on} track{'s' if n_on > 1 else ''}",
                              fontsize=7, color=INK, labelpad=3, linespacing=1.1)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.92, bottom=0.01, wspace=0.04, hspace=0.05)
    save(fig, "fig_tracking_strip.png")


# ------------------------------------------------------------------ figure C: failures

def fig_failure_cases() -> None:
    man = manifest().set_index("video")
    fold = lambda v: int(man.loc[v, "fold"])  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 512 / 640 + 0.55))
    title = TITLE                      # short titles: the caption carries each panel's full story

    # (a) detected, but mostly below the start threshold: of the bird clips that some tracker never
    # follows (oracle policy), the one with the fewest detections >= the start threshold; shown at
    # its median-confidence detection on the bird. (V_BIRD_016 until v2; the v3 detectors track it.)
    drop = pd.read_csv(RES / "tracker_dropped_clips.csv").query("policy == 'oracle' and cls == 'BIRD'")
    v = drop.sort_values(["dets_ge_0.45", "video"]).video.iloc[0]
    n_lost = int((drop.video == v).sum())
    n_hi = int(drop.loc[drop.video == v, "dets_ge_0.45"].iloc[0])
    g, D = gt_boxes(v), dets(v, fold(v))
    on = sorted((float(d[5]), int(d[0]), xywh(d)) for d in D
                if int(d[0]) in g and tf.iou(g[int(d[0])], xywh(d)) >= 0.3)
    conf, f, db = on[len(on) // 2]
    gb = g[f]
    print(f"  (a) {v}: {n_hi} detections >= {CONF_START} in the clip, no track on the bird for {n_lost} "
          f"trackers (oracle); {len(on)} detections on the bird, median {conf:.3f} at frame {f}")
    full_with_zoom(axes[0, 0], frame(v, f), gb,
                   [(gb, {"colour": GT_EDGE, "dashed": True}),
                    (db, {"colour": DET, "label": f"{conf:.2f}"})], half=32)
    axes[0, 0].set_title(f"(a) {v}, frame {f}\ndetected at {conf:.2f}; only {n_hi} "
                         f"reach {CONF_START:.2f}", **title)

    # (b) a correctly labelled bird the detector misses
    birds = [v for v in man.index if man.loc[v, "class"] == "BIRD" and len(gt_boxes(v)) >= MIN_FRAMES]
    rec = {v: detector_recall(v, fold(v)) for v in birds}
    miss = min(rec, key=rec.get)
    g, Dm = gt_boxes(miss), dets(miss, fold(miss))
    empty = [f for f in sorted(g) if not any(np.linalg.norm(
        np.array([d[1] + d[3], d[2] + d[4]]) / 2 - np.array([g[f][0] + g[f][2] / 2, g[f][1] + g[f][3] / 2]))
        < 1.5 * np.hypot(g[f][2], g[f][3]) for d in Dm[Dm[:, 0] == f])]
    f = empty[len(empty) // 2]
    print(f"  (b) lowest detector recall among usable bird clips: {miss} ({rec[miss]:.2f}); "
          f"{len(empty)}/{len(g)} labelled frames with no detection nearby; shown at frame {f}")
    boxes = [(g[f], {"colour": GT_EDGE, "dashed": True})] + [(xywh(d), {"colour": DET}) for d in Dm[Dm[:, 0] == f]]
    full_with_zoom(axes[0, 1], frame(miss, f), g[f], boxes, half=32)
    axes[0, 1].set_title(f"(b) {miss}, frame {f}\nno detection; clip recall {rec[miss]:.2f}",
                         **title)

    # (c) a source label removed by the audit: birds beside an empty box
    v = "V_BIRD_020"
    go, Dv = gt_boxes(v, GT_ORIG), dets(v, fold(v))
    best = None
    for f in sorted(go):
        c0 = np.array([go[f][0] + go[f][2] / 2, go[f][1] + go[f][3] / 2])
        for d in Dv[Dv[:, 0] == f]:
            dist = float(np.linalg.norm(np.array([d[1] + d[3], d[2] + d[4]]) / 2 - c0))
            if tf.iou(go[f], xywh(d)) == 0 and (best is None or dist < best[0]):
                best = (dist, f, xywh(d))
    dist, f, db = best
    x0, y0 = min(go[f][0], db[0]), min(go[f][1], db[1])
    x1, y1 = max(go[f][0] + go[f][2], db[0] + db[2]), max(go[f][1] + go[f][3], db[1] + db[3])
    union = (x0, y0, x1 - x0, y1 - y0)
    print(f"  (c) {v} source label at frame {f}: nearest detection outside it {dist:.0f} px away")
    full_with_zoom(axes[1, 0], frame(v, f), union,
                   [(go[f], {"colour": GT_EDGE, "dashed": True}), (db, {"colour": DET})],
                   half=max(40.0, 0.65 * max(union[2], union[3])))
    axes[1, 0].set_title(f"(c) {v}, frame {f}\nsource label (dashed) on empty sky", **title)

    # (d) the same clip: the longest track follows another bird of the flock
    tk = tf.read_mot(RES / "tracks" / "bytetrack" / f"{v}.txt")
    lid = tf.match_clip(v, tk, 0.3)["longest_id"]
    rows = sorted((fr, t) for fr in tk for t in tk[fr] if t[0] == lid)
    fr, t = rows[len(rows) // 2]
    print(f"  (d) {v}: ByteTrack longest #{lid} frames {rows[0][0]}-{rows[-1][0]} ({len(rows)} rows); frame {fr}")
    full_with_zoom(axes[1, 1], frame(v, fr), t[1:], [(t[1:], {"colour": ID_COLOURS[0], "label": f"#{lid}"})], half=32)
    axes[1, 1].set_title(f"(d) {v}, frame {fr}\nlongest ByteTrack track #{lid}: another bird", **title)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.925, bottom=0.005, wspace=0.03, hspace=0.2)
    save(fig, "fig_failure_cases.png")


FIGS = {"dataset": fig_dataset_examples, "strip": fig_tracking_strip, "failure": fig_failure_cases}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated: " + ",".join(FIGS))
    args = ap.parse_args()
    style()
    for name in [x for x in args.only.split(",") if x] or list(FIGS):
        print(f"figure {name}")
        FIGS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
