#!/usr/bin/env python3
"""Draw ground-truth boxes on sampled frames so a human can check the labels.

DDS drones are often under 32x32 px in a 4K frame, so a downscaled full frame proves
nothing. Each sample is therefore a pair: the full frame (context, boxes marked) beside
a zoomed crop around the box. A contact sheet of all samples is written at the end.

Frames are pulled with ffmpeg's select filter, which decodes from the start and counts
decoded frames exactly. cv2's frame seek is NOT used: six of these MP4s carry leading
frames that players discard, and a sloppy seek would fake a misalignment (see DAI c10b7ccf).

Usage
    python src/paper1/preview_gt.py                  # 16 random videos, 1 frame each
    python src/paper1/preview_gt.py --n 8 --seed 3
    python src/paper1/preview_gt.py --video two_distant_phantom --n 4
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

GREEN = (80, 220, 80)
YELLOW = (60, 220, 240)
PANEL_H = 420            # height of each half-panel in the sheet


def read_gt(gt_file: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for line in gt_file.read_text().splitlines():
        if not line.strip():
            continue
        f, tid, x, y, w, h = line.split(",")[:6]
        frames.setdefault(int(f), []).append((int(tid), float(x), float(y), float(w), float(h)))
    return frames


def grab_frame(video: Path, index0: int, out_png: Path) -> bool:
    """Exact decoded-frame extraction (0-based index)."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video),
           "-vf", f"select=eq(n\\,{index0})", "-frames:v", "1", str(out_png)]
    subprocess.run(cmd, check=False)
    return out_png.is_file() and out_png.stat().st_size > 0


def fit(img: np.ndarray, height: int) -> np.ndarray:
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), height),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)


def caption(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((26, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text[:120], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def make_panel(frame: np.ndarray, boxes, video: str, frame_no: int) -> np.ndarray:
    """Full frame with every box marked, beside a zoom on the first box."""
    full = frame.copy()
    for tid, x, y, w, h in boxes:
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(full, p1, p2, GREEN, max(2, frame.shape[1] // 640))
        cv2.putText(full, f"id{tid}", (p1[0], max(12, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, frame.shape[1] / 2200), YELLOW, 2, cv2.LINE_AA)

    tid, x, y, w, h = boxes[0]
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * 4
    half = max(half, 60)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(frame.shape[1], cx + half)), int(min(frame.shape[0], cy + half))
    crop = frame[y0:y1, x0:x1].copy()
    if crop.size:
        cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), GREEN, 1)
        crop = fit(crop, PANEL_H)
    else:
        crop = np.zeros((PANEL_H, PANEL_H, 3), dtype=np.uint8)

    left = caption(fit(full, PANEL_H), f"{video}  frame {frame_no}  {frame.shape[1]}x{frame.shape[0]}")
    right = caption(crop, f"zoom x{2 * half / max(w, h):.0f}  box {w:.0f}x{h:.0f}px  {len(boxes)} obj")
    gap = np.full((left.shape[0], 8, 3), 40, dtype=np.uint8)
    return np.hstack([left, gap, right])


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=root / "gt" / "dds_drone")
    ap.add_argument("--videos", type=Path,
                    default=root / "wosdetc_train_videos" / "wosdetc_train_videos")
    ap.add_argument("--out", type=Path, default=root / "data_checks" / "gt_preview")
    ap.add_argument("--n", type=int, default=16, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=str, default=None, help="sample only this video")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_stem = {p.stem: p for p in args.videos.iterdir() if p.is_file()}
    seqs = sorted(d.name for d in args.gt.iterdir() if (d / "gt" / "gt.txt").is_file())
    if args.video:
        seqs = [s for s in seqs if s == args.video] or seqs
        picks = [(args.video, None)] * args.n
    else:
        picks = [(s, None) for s in rng.sample(seqs, min(args.n, len(seqs)))]

    args.out.mkdir(parents=True, exist_ok=True)
    panels, missing = [], []

    with tempfile.TemporaryDirectory() as tmp:
        for i, (seq, _) in enumerate(picks, 1):
            if seq not in by_stem:
                missing.append(f"{seq}: no video file")
                continue
            frames = read_gt(args.gt / seq / "gt" / "gt.txt")
            if not frames:
                missing.append(f"{seq}: empty gt")
                continue
            frame_no = rng.choice(sorted(frames))            # 1-based (MOT)
            png = Path(tmp) / f"{i}.png"
            if not grab_frame(by_stem[seq], frame_no - 1, png):
                missing.append(f"{seq}: could not decode frame {frame_no}")
                continue
            img = cv2.imread(str(png))
            if img is None:
                missing.append(f"{seq}: frame {frame_no} unreadable")
                continue
            panel = make_panel(img, frames[frame_no], seq, frame_no)
            cv2.imwrite(str(args.out / f"{seq}_f{frame_no}.jpg"), panel,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            panels.append(panel)
            print(f"  {i:2d}/{len(picks)} {seq} frame {frame_no} "
                  f"({len(frames[frame_no])} box(es))")

    if panels:
        width = max(p.shape[1] for p in panels)
        padded = [np.pad(p, ((0, 0), (0, width - p.shape[1]), (0, 0))) for p in panels]
        sheet = np.vstack(padded)
        sheet_path = args.out / "contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\n{len(panels)} panels -> {args.out}\ncontact sheet -> {sheet_path}")
    for m in missing:
        print("  SKIP", m, file=sys.stderr)
    return 0 if panels else 1


if __name__ == "__main__":
    raise SystemExit(main())
