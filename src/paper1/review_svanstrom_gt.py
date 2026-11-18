#!/usr/bin/env python3
"""Review sheets for Svanström ground truth that drifts off its target (docs/errors.md 2026-09-24).

Two automatic ways of finding drifted frames were tried and rejected (ADR-006): a detection
follower, which could not tell a detector miss from a drifted label and dropped correct boxes
on drone clips, and a box-content contrast test, which scored many correct boxes on faint
25-px targets as empty. Drift is therefore judged BY EYE.

This script renders, for each clip, every STEP-th labelled frame as a nearest-neighbour zoom
around the GT box (thin white). A reviewer marks the frame ranges whose box holds no bird or
drone. Those ranges go into gt/svanstrom_corrections.csv as `drop` rows, and
build_svanstrom_subset.py removes them. Boxes are only removed, never redrawn, so a
correction can never place a box on the wrong object.

    python src/paper1/review_svanstrom_gt.py --videos V_BIRD_017,V_BIRD_020
    python src/paper1/review_svanstrom_gt.py --videos V_BIRD_021 --step 2 --from 140 --to 230
    python src/paper1/review_svanstrom_gt.py --check     # re-draw every clip listed in corrections, kept frames only

Writes data_checks/svanstrom_label_audit/review/<video>[_<from>-<to>].jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracker_features as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "gt" / "svanstrom_visible"
IMAGES = ROOT / "datasets" / "svan_det" / "images"
CORR = ROOT / "gt" / "svanstrom_corrections.csv"
OUT = ROOT / "data_checks" / "svanstrom_label_audit" / "review"
HALF, TILE, COLS = 48, 216, 8          # 96-px window shown at 216 px (x2.25)


def sheet(video: str, frames: list[int], G: dict, title: str) -> np.ndarray:
    tiles = []
    for f in frames:
        img = cv2.imread(str(IMAGES / video / f"{f:06d}.jpg"))
        H, W = img.shape[:2]
        x, y, w, h = G[f]
        cx, cy = x + w / 2, y + h / 2
        x0 = int(min(max(cx - HALF, 0), W - 2 * HALF))
        y0 = int(min(max(cy - HALF, 0), H - 2 * HALF))
        crop = img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].copy()
        t = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        s = TILE / (2 * HALF)
        cv2.rectangle(t, (int((x - x0) * s), int((y - y0) * s)), (int((x + w - x0) * s), int((y + h - y0) * s)),
                      (255, 255, 255), 1)
        cv2.putText(t, f"f{f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % COLS:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    bar = np.zeros((26, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, grid])


def gt_of(video: str) -> dict[int, tuple]:
    return {f: b[0][1:] for f, b in tf.read_mot(GT_DIR / video / "gt" / "gt.txt").items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10 ** 6)
    ap.add_argument("--check", action="store_true",
                    help="draw only the frames KEPT after gt/svanstrom_corrections.csv, for every clip it lists")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check:
        corr = pd.read_csv(CORR, comment="#")
        for v, c in corr.groupby("video"):
            G = gt_of(v)                                # gt.txt already has the corrections applied
            fs = sorted(G)[::args.step]
            cv2.imwrite(str(OUT / f"{v}_kept.jpg"), sheet(v, fs, G, f"{v}  kept frames after corrections, "
                        f"every {args.step}th of {len(G)}"), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  {v}: {len(G)} labelled frames kept -> {OUT / (v + '_kept.jpg')}")
        return 0
    for v in [x for x in args.videos.split(",") if x]:
        G = gt_of(v)
        fs = [f for f in sorted(G) if args.lo <= f <= args.hi][::args.step]
        tag = "" if (args.lo, args.hi) == (0, 10 ** 6) else f"_{args.lo}-{args.hi}"
        path = OUT / f"{v}{tag}.jpg"
        cv2.imwrite(str(path), sheet(v, fs, G, f"{v}  every {args.step}th labelled frame "
                    f"({len(G)} labelled, span {min(G)}-{max(G)})"), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {v}: {len(fs)} tiles -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
