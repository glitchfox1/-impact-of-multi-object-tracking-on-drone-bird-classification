#!/usr/bin/env python3
"""Decode the 140 selected Svanstrom clips into frames + YOLO labels.

Why this exists: the ceiling test (feature_ceiling.py) ran on PERFECT ground-truth
tracks. To find out what the 7 features are worth on REAL tracks we need a detector,
and a detector needs images. This writes them once so every later stage -- detector
training, detection caching, all 10 boxmot trackers -- reads identical pixels.

The detector is deliberately CLASS-AGNOSTIC (single class "flying object"). It must
not know bird from drone: that decision belongs to the track features under test.
Letting the detector classify would answer a different, easier question.

Layout written under datasets/svan_det/:
    images/<video>/<frame:06d>.jpg      every frame (trackers need continuity)
    labels/<video>/<frame:06d>.txt      YOLO box, class 0, empty when unlabelled
    folds/fold<k>_{train,val}.txt       per-fold image lists, training stride applied.
                                        val = the held-out fold; run_trackers reads its
                                        clip names, so it always lists every held-out clip
    folds/fold<k>_{itrain,ival,test}.txt  detector training lists (ADR-007): the training
                                        clips split by CLIP into itrain / ival (inner
                                        validation, stratified by class, seeded), and the
                                        held-out clips as test. The 6 clips corrected under
                                        ADR-006 contribute only their labelled frames to all
                                        three: their unlabelled frames often show a bird, so
                                        as background images they would teach the detector to
                                        ignore birds.
    folds/inner_val_clips.csv           which clips form each fold's inner validation set

--labels-only rewrites labels/ (and the lists) from the current ground truth without
decoding any video -- the way to apply gt/svanstrom_corrections.csv to the training labels.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_svanstrom_subset import load_corrections  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "gt" / "svanstrom_visible" / "_manifest.csv"
CORRECTIONS = ROOT / "gt" / "svanstrom_corrections.csv"
VIDEO_DIR = ROOT / "svanstrom_dataset" / "Data" / "Video_V"
OUT = ROOT / "datasets" / "svan_det"


def load_gt(seq_dir: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """MOT gt.txt -> {frame: [(x, y, w, h), ...]}, 1-based frames as written."""
    boxes: dict[int, list] = defaultdict(list)
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        f, _tid, x, y, w, h = line.split(",")[:6]
        boxes[int(f)].append((float(x), float(y), float(w), float(h)))
    return boxes


def to_yolo(x: float, y: float, w: float, h: float, W: int, H: int) -> str | None:
    """MOT top-left xywh -> YOLO normalised cx cy w h, clipped to the frame."""
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    bw, bh = (x1 - x0) / W, (y1 - y0) / H
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export_clip(video: str, seq_dir: Path, jpeg_q: int) -> tuple[int, int]:
    src = VIDEO_DIR / f"{video}.mp4"
    if not src.exists():
        raise FileNotFoundError(src)
    img_dir, lab_dir = OUT / "images" / video, OUT / "labels" / video
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(seq_dir)
    cap = cv2.VideoCapture(str(src))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_img = n_box = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1                                    # 1-based, matches gt.txt
        cv2.imwrite(str(img_dir / f"{idx:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        lines = [s for b in gt.get(idx, []) if (s := to_yolo(*b, W, H))]
        (lab_dir / f"{idx:06d}.txt").write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)
    cap.release()
    return n_img, n_box


def export_labels(video: str, seq_dir: Path, W: int, H: int) -> tuple[int, int]:
    """Labels only, for the frames already on disk -- same writer as export_clip."""
    frames = sorted((OUT / "images" / video).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no exported frames for {video}; run a full export first")
    lab_dir = OUT / "labels" / video
    lab_dir.mkdir(parents=True, exist_ok=True)
    gt = load_gt(seq_dir)
    n_box = 0
    for p in frames:
        idx = int(p.stem)
        lines = [s for b in gt.get(idx, []) if (s := to_yolo(*b, W, H))]
        (lab_dir / f"{idx:06d}.txt").write_text("\n".join(lines))
        n_box += len(lines)
    return len(frames), n_box


def strided(videos, stride: int, corrected: set[str] = frozenset()) -> list[str]:
    """Every `stride`-th frame of each clip. A `corrected` clip (one named in
    gt/svanstrom_corrections.csv) then keeps only its LABELLED frames: its labels are known
    to be wrong somewhere, so an empty label there does not mean "no bird" -- in V_BIRD_020
    a flock is visible in every frame and no frame is labelled. Filtering after striding
    keeps the frames a subset of the old list's."""
    paths = []
    for v in sorted(videos):
        for p in sorted((OUT / "images" / v).glob("*.jpg"))[::stride]:
            if v in corrected and not (OUT / "labels" / v / f"{p.stem}.txt").stat().st_size:
                continue
            paths.append(str(p))
    return paths


def write_fold_lists(per_fold: dict[str, list[str]], cls_of: dict[str, str], stride: int,
                     ival_frac: float, seed: int) -> None:
    """Train frames are strided (consecutive frames are near-duplicates and only slow
    training down); val keeps the same stride so the number is readable."""
    folds_dir = OUT / "folds"
    folds_dir.mkdir(exist_ok=True)
    corrected = set(load_corrections(CORRECTIONS))
    ival_rows = []
    for k in sorted(per_fold):
        val_vids = set(per_fold[k])
        train_vids = set().union(*per_fold.values()) - val_vids
        # The inner validation set is drawn by CLIP, never by frame (frames of one clip are
        # near-duplicates), per class so every class is present, from a seed fixed per fold.
        rng = random.Random(seed + int(k))
        ival = set()
        for c in sorted({cls_of[v] for v in train_vids}):
            pool = sorted(v for v in train_vids if cls_of[v] == c)
            ival |= set(rng.sample(pool, max(1, round(len(pool) * ival_frac))))
        ival_rows += [{"fold": k, "video": v, "class": cls_of[v]} for v in sorted(ival)]
        lists = {
            "train": strided(train_vids, stride),          # the original lists, unchanged
            "val": strided(val_vids, stride),
            "itrain": strided(train_vids - ival, stride, corrected),
            "ival": strided(ival, stride, corrected),
            "test": strided(val_vids, stride, corrected),
        }
        for split, paths in lists.items():
            (folds_dir / f"fold{k}_{split}.txt").write_text("\n".join(paths))
            n_clips = len({Path(p).parent.name for p in paths})
            print(f"fold {k} {split:6s}: {n_clips:3d} clips, {len(paths):6d} frames")
    with (folds_dir / "inner_val_clips.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fold", "video", "class"])
        w.writeheader()
        w.writerows(ival_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3,
                    help="training-frame stride; val and tracking always use every frame")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N clips")
    ap.add_argument("--labels-only", action="store_true",
                    help="rewrite labels + lists from the current GT; do not decode videos")
    ap.add_argument("--skip-existing-images", action="store_true",
                    help="decode only clips whose frames are not on disk yet (e.g. clips added by "
                         "build_svanstrom_subset.py --extend); labels are rewritten for all")
    ap.add_argument("--ival-frac", type=float, default=1 / 7,
                    help="share of each class's training clips held out for inner validation")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(MANIFEST.open()) if r["selected"] == "1"]
    if args.limit:
        rows = rows[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    per_fold: dict[str, list[str]] = defaultdict(list)
    cls_of: dict[str, str] = {}
    total_img = total_box = 0
    for i, r in enumerate(rows, 1):
        video, fold = r["video"], r["fold"]
        seq_dir = ROOT / "gt" / "svanstrom_visible" / video
        on_disk = len(list((OUT / "images" / video).glob("*.jpg")))
        if args.labels_only or (args.skip_existing_images and on_disk == int(r["frames"])):
            n_img, n_box = export_labels(video, seq_dir, int(r["width"]), int(r["height"]))
        else:
            n_img, n_box = export_clip(video, seq_dir, args.jpeg_quality)
        if n_img != int(r["frames"]):
            print(f"  ! {video}: {n_img} frames, manifest says {r['frames']}")
        if n_box != int(r["boxes_written"]):
            print(f"  ! {video}: wrote {n_box} boxes, manifest says {r['boxes_written']}")
        per_fold[fold].append(video)
        cls_of[video] = r["class"]
        total_img += n_img
        total_box += n_box
        print(f"[{i:3d}/{len(rows)}] {video:22s} fold {fold}  {n_img:4d} frames  {n_box:4d} boxes",
              flush=True)

    write_fold_lists(per_fold, cls_of, args.stride, args.ival_frac, args.seed)
    print(f"\nTOTAL {total_img} frames, {total_box} boxes -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
