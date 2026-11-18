#!/usr/bin/env python3
"""Stage 2 of the tracker arm: cache detections once, then replay them into every tracker.

The comparison is only fair if every tracker sees byte-identical detections, so the
detector runs ONCE per fold and its output is cached to disk. Each tracker then replays
that cache. Any difference in the MOT output is therefore the tracker's doing and
nothing else -- not detector nondeterminism, not a different confidence threshold.

Detections are cached at a deliberately LOW confidence (default 0.10). Thresholding is
part of what a tracker does (the ByteTrack idea is precisely the low-confidence second
pass), so cutting high here would quietly disable that and hand the comparison to the
wrong stage.

Writes
    results/dets/fold<k>/<video>.npy     (frame, x1, y1, x2, y2, conf) float32
    results/tracks/<tracker>/<video>.txt MOT: frame,id,x,y,w,h,conf,-1,-1,-1
    results/tracks/_timing.csv           per tracker/clip wall time and track count,
                                         MERGED across runs on (tracker, fold, video)

Resumable: a clip whose track file already exists is skipped, exactly as a cached
detection file is. Pass --overwrite to redo everything -- REQUIRED after any change to
the tracker configuration (CONF_HIGH / CONF_LOW, aliases) or to the cached detections,
or the old tracks are silently kept.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "svan_det"
RES = ROOT / "results"
TIMING_CSV = RES / "tracks" / "_timing.csv"
TIMING_FIELDS = ["tracker", "fold", "video", "frames", "dets", "rows", "ids", "seconds", "fps", "device"]
TIMING_KEY = ("tracker", "fold", "video")

# sam2mot is deliberately excluded: Sam2Mot.update() raises "Sam2Mot requires full-frame
# detection masks" and this pipeline produces boxes. Including it would mean running SAM2
# over every frame to turn boxes into masks -- a second model and a different experiment.
# The comparison is therefore over the 9 BOX trackers boxmot 25 ships. See docs/errors.md.
ALL_TRACKERS = ["bytetrack", "ocsort", "sfsort", "occluboost", "botsort",
                "strongsort", "deepocsort", "hybridsort", "boosttrack"]
CLASS_MAP = {"bytetrack": "ByteTrack", "ocsort": "OcSort", "sfsort": "SFSORT",
             "occluboost": "OccluBoost", "botsort": "BotSort", "strongsort": "StrongSort",
             "deepocsort": "DeepOcSort", "hybridsort": "HybridSort",
             "boosttrack": "BoostTrack", "sam2mot": "Sam2Mot"}


def fold_videos(fold: int) -> list[str]:
    lines = (DATA / "folds" / f"fold{fold}_val.txt").read_text().split()
    return sorted({Path(p).parent.name for p in lines})


def frames_of(video: str) -> list[Path]:
    return sorted((DATA / "images" / video).glob("*.jpg"))


def cache_detections(fold: int, conf: float, imgsz: int, batch: int,
                     weights_dir: Path = ROOT / "runs" / "detect", out_root: Path = RES,
                     videos: list[str] | None = None) -> None:
    """Run the fold's detector over every frame of its held-out clips, once."""
    from ultralytics import YOLO

    weights = weights_dir / f"fold{fold}" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"no detector for fold {fold}: {weights}")
    model = YOLO(str(weights))
    out_dir = out_root / "dets" / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for vi, video in enumerate(videos or fold_videos(fold), 1):
        dst = out_dir / f"{video}.npy"
        if dst.exists():
            print(f"[fold {fold}] {vi:3d} {video:22s} cached, skip", flush=True)
            continue
        frames = frames_of(video)
        rows = []
        for s in range(0, len(frames), batch):
            chunk = frames[s: s + batch]
            preds = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz,
                                  device=0, verbose=False)
            for off, r in enumerate(preds):
                fno = int(chunk[off].stem)          # 1-based, matches gt.txt
                if len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cf = r.boxes.conf.cpu().numpy()
                    for b, c in zip(xyxy, cf):
                        rows.append((fno, b[0], b[1], b[2], b[3], c))
        arr = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), np.float32)
        np.save(dst, arr)
        print(f"[fold {fold}] {vi:3d} {video:22s} {len(frames):4d} frames -> "
              f"{len(arr):5d} dets", flush=True)


# A SHARED CONFIDENCE OPERATING POINT, imposed on every tracker.
#
# boxmot's defaults assume a COCO-scale detector whose confident detections sit at 0.9+.
# This detector is a small-object detector: measured on the fold-0 cache its confidences
# run min 0.10, median 0.48, p95 0.78, max 0.87, and only 13% clear 0.70. Left on their
# defaults, SFSORT (new_track_th 0.70) and OccluBoost/BotSort (new_track_thresh 0.60) are
# starved of track initialisations and emit almost nothing -- SFSORT produced 0 rows on
# most clips in the first attempt. That measures a threshold mismatch, not tracking skill.
#
# So every tracker is pinned to the same operating point, expressed in the detector's own
# confidence scale: START (a track may only be started by a detection >= HIGH) at the
# detector's median, FLOOR (detections down to LOW may continue an existing track) at the
# caching threshold. Trackers then differ only in ASSOCIATION, which is what the comparison
# is about. Each tracker spells these differently, hence the alias table; unknown names are
# simply not passed.
CONF_HIGH = 0.45          # ~detector median; ByteTrack's own default, used as the reference
CONF_LOW = 0.10           # matches --conf used when caching detections
HIGH_ALIASES = ("track_thresh", "high_th", "new_track_th", "new_track_thresh",
                "track_high_thresh", "det_thresh")
LOW_ALIASES = ("min_conf", "low_th", "low_thresh", "track_low_thresh")

# det_thresh. BoxMOT 25's base class only STORES it (default 0.30); each tracker reads it
# itself, and what it means differs per tracker (read from the 25.0.0 source, 2026-09-25):
#   ocsort, deepocsort, boosttrack  start threshold (and ocsort's high/low split)  -> HIGH
#   hybridsort                      high/low split; tracks start only from the high
#                                   side (its track_thresh only bounds the TCM
#                                   confidence scores)                              -> HIGH
#   occluboost                      matching floor; new_track_thresh is the start   -> LOW
#   botsort, strongsort             never read
#   bytetrack, sfsort               rejected; derived from track_thresh / high_th
# The default rule below (LOW when a named high threshold exists, else HIGH) is right for all
# of them except hybridsort, which it set to 0.10 -- so 71% of Hybrid-SORT tracks started
# below 0.45 (docs/errors.md 2026-09-24 ConfigError). The rule's original rationale, that the
# base class filters on det_thresh before the named thresholds, was wrong: it does not.
DET_THRESH_OVERRIDE = {"hybridsort": CONF_HIGH}
# Low pass switched off by default. OC-SORT has the BYTE second association but ships with
# use_byte=False, so its 0.10 floor was configured and never used.
EXTRA_KWARGS = {"ocsort": {"use_byte": True}}
# StrongSORT has no start threshold at all: any unmatched detection >= min_conf starts a
# (tentative) track. It gets a start gate; min_conf stays the floor.
START_GATED = {"strongsort"}
# Not changed, by design of the algorithm: deepocsort and boosttrack have no low-confidence
# pass (they use only detections >= HIGH); boosttrack and occluboost raise ("boost") some
# low confidences before thresholding, and the start threshold applies to the boosted value.


def gate_track_starts(tracker, start: float = CONF_HIGH) -> None:
    """Start a StrongSORT track only from a detection with conf >= `start`.

    Wraps the instance's _initiate_track, the one place StrongSORT (and its project copy)
    creates tracks. Detections below `start` still update tracks that already exist."""
    initiate = tracker._initiate_track

    def gated(detection):
        if detection.conf >= start:
            initiate(detection)

    tracker._initiate_track = gated


def construct_tracker(name: str, cls, fps: float, **extra):
    """Build `cls` at the shared operating point. Shared by build_tracker and the Phase 4
    copies (behaviour_trackers.run_behaviour_trackers), so the two can never drift apart;
    `extra` carries the copies' own options (behaviour cues, device)."""
    # Trackers disagree on which knobs exist; pass only what each explicitly accepts.
    # Every one of them also takes **kwargs, so a name check is the only real guard.
    params = set(inspect.signature(cls.__init__).parameters)
    kw = {}
    if "frame_rate" in params:
        kw["frame_rate"] = int(round(fps))
    for a in HIGH_ALIASES:
        if a in params:
            kw[a] = CONF_HIGH
    for a in LOW_ALIASES:
        if a in params:
            kw[a] = CONF_LOW
    kw.update(EXTRA_KWARGS.get(name, {}))
    kw.update(extra)
    det = DET_THRESH_OVERRIDE.get(
        name, CONF_LOW if any(a in kw for a in HIGH_ALIASES) else CONF_HIGH)
    # ByteTrack and SFSORT reject det_thresh outright, hence the try -- never leave a
    # tracker unconstructed over a tuning detail.
    try:
        tracker = cls(**kw, det_thresh=det)
    except TypeError:
        tracker = cls(**kw)
    if name in START_GATED:
        gate_track_starts(tracker)
    return tracker


def build_tracker(name: str, fps: float, device: str = "cpu"):
    import boxmot
    cls = getattr(boxmot, CLASS_MAP[name])
    # Only trackers with a ReID network take a device; CPU is BoxMOT's default, so pass nothing then.
    extra = {"device": device} if device != "cpu" and getattr(cls, "accepts_embeddings", False) else {}
    return construct_tracker(name, cls, fps, **extra)


def resolve_device(arg: str) -> str:
    """'auto' -> '0' (first GPU) when CUDA is available, else 'cpu'; anything else unchanged."""
    if arg != "auto":
        return arg
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def track_clip(tracker, frames: list[Path], by_frame: dict[int, np.ndarray]) -> tuple[list[str], float]:
    """Replay one clip's cached detections through `tracker`; MOT lines and wall time."""
    import cv2
    lines, t0 = [], time.perf_counter()
    for p in frames:
        fno = int(p.stem)
        img = cv2.imread(str(p))
        d = by_frame.get(fno, np.zeros((0, 6), np.float32))
        res = tracker.update(d, img)
        res = np.asarray(res if res is not None else np.zeros((0, 8)))
        if res.ndim != 2 or not len(res):
            continue
        for r in res:
            x1, y1, x2, y2 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            tid = int(r[4])
            cf = float(r[5]) if len(r) > 5 else 1.0
            lines.append(f"{fno},{tid},{x1:.2f},{y1:.2f},"
                         f"{x2 - x1:.2f},{y2 - y1:.2f},{cf:.4f},-1,-1,-1")
    return lines, time.perf_counter() - t0


def run_tracker(name: str, fold: int, videos: list[str], timing: list,
                overwrite: bool = False, out_root: Path = RES, device: str = "cpu") -> None:
    out_dir = out_root / "tracks" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        dst = out_dir / f"{video}.txt"
        # Resume. The output below is written temp-then-replace, so an existing file is
        # always a finished clip -- including a legitimate 0-byte file for a clip where
        # the detector found nothing, which must not be redone either.
        if dst.exists() and not overwrite:
            print(f"  {name:11s} {video:22s} done, skip", flush=True)
            continue
        dets = np.load(out_root / "dets" / f"fold{fold}" / f"{video}.npy")
        frames = frames_of(video)
        by_frame: dict[int, np.ndarray] = {}
        if len(dets):
            for fno in np.unique(dets[:, 0]).astype(int):
                d = dets[dets[:, 0] == fno][:, 1:6]
                by_frame[fno] = np.hstack([d, np.zeros((len(d), 1), np.float32)])

        try:
            tracker = build_tracker(name, 30.0, device)
            lines, dt = track_clip(tracker, frames, by_frame)
        except Exception as e:
            # The GPU ReID path was never exercised on this machine before the 2026-09-26
            # rerun, which runs unattended: a device failure (CUDA OOM included) must cost
            # speed, not the night. The clip is redone from scratch on the CPU and the tracker
            # stays there; the timing row records where each clip actually ran.
            if device == "cpu":
                raise
            print(f"  !! {name} {video}: GPU run failed ({type(e).__name__}: {e}) -- "
                  f"redoing on CPU; {name} stays on CPU for the rest of this fold", flush=True)
            device = "cpu"
            tracker = build_tracker(name, 30.0, device)
            lines, dt = track_clip(tracker, frames, by_frame)
        ran_on = device if getattr(tracker, "generates_embeddings", False) else "n/a (no ReID)"
        # A kill mid-write must never leave a truncated file that the resume check above
        # would then trust as finished.
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text("\n".join(lines))
        replace_retrying(tmp, dst)
        n_ids = len({int(l.split(",")[1]) for l in lines}) if lines else 0
        timing.append({"tracker": name, "fold": fold, "video": video,
                       "frames": len(frames), "dets": len(dets), "rows": len(lines),
                       "ids": n_ids, "seconds": round(dt, 2),
                       "fps": round(len(frames) / dt, 1) if dt > 0 else 0, "device": ran_on})
        print(f"  {name:11s} {video:22s} {len(lines):5d} rows  {n_ids:3d} ids  "
              f"{len(frames) / dt:5.1f} fps  [{ran_on}]", flush=True)
        # Per clip, the same granularity as the track file: resume skips any clip that has
        # a track file, so a timing row not saved alongside it would be lost for good.
        save_timing(timing, out_root / "tracks" / "_timing.csv")


def replace_retrying(src: Path, dst: Path, tries: int = 5) -> None:
    """src.replace(dst), retried briefly. On Windows a virus scanner can hold a file it
    has just seen written, and the rename then fails with PermissionError for a moment."""
    for i in range(tries):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.2 * (i + 1))


def save_timing(pending: list[dict], timing_csv: Path = TIMING_CSV) -> None:
    """Merge `pending` into _timing.csv -- one row per (tracker, fold, video), the newest
    wins -- then empty `pending`.

    Rewriting the file with only the current run's rows (the old behaviour) erased every
    earlier run: finishing folds 1-4 would have wiped the fold-0 timings.

    Timing is bookkeeping and must never stop the sweep. If the write fails (_timing.csv
    open in Excel, a virus scanner holding it) the rows stay in `pending` and go out with
    the next save. Temp-then-replace, so a kill mid-write leaves the previous file intact;
    the temp name carries the PID so two processes never write the same temp file. Two
    sweeps at once are still not safe: a row can be lost in the read-merge-write window.
    """
    if not pending:
        return
    tmp = timing_csv.with_name(f"{timing_csv.name}.{os.getpid()}.tmp")
    try:
        merged: dict[tuple, dict] = {}
        if timing_csv.exists():
            with timing_csv.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    merged[tuple(str(r[k]) for k in TIMING_KEY)] = r
        for r in pending:
            merged[tuple(str(r[k]) for k in TIMING_KEY)] = r
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TIMING_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(merged.values(),
                               key=lambda r: (r["tracker"], int(r["fold"]), r["video"])))
        replace_retrying(tmp, timing_csv)
    except Exception as e:                          # bookkeeping must not kill tracking
        print(f"  !! timing not saved ({type(e).__name__}: {e}) -- "
              f"{len(pending)} row(s) kept for the next save", flush=True)
        return
    pending.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--trackers", default=",".join(ALL_TRACKERS))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo clips whose track file already exists (default: skip them)")
    ap.add_argument("--out", type=Path, default=RES,
                    help="results root for dets/ and tracks/ (default results/; test runs elsewhere)")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detect",
                    help="detector runs dir holding fold<k>/weights/best.pt")
    ap.add_argument("--videos", default="",
                    help="comma list: only these clips of the chosen folds (test runs)")
    ap.add_argument("--device", default="auto",
                    help="ReID network device: auto (GPU if available) | cpu | CUDA index. Before "
                         "2026-09-26 every ReID tracker ran on the CPU (BoxMOT's default)")
    args = ap.parse_args()
    device = resolve_device(args.device)
    if device != "cpu":
        import torch
        torch.backends.cudnn.deterministic = True     # reproducible reruns, as in Phase 4
        torch.backends.cudnn.benchmark = False
    print(f"ReID device: {device}", flush=True)

    folds = [int(f) for f in args.folds.split(",")]
    only = set(filter(None, args.videos.split(",")))
    clips = {k: [v for v in fold_videos(k) if not only or v in only] for k in folds}
    for fold in folds:
        cache_detections(fold, args.conf, args.imgsz, args.batch,
                         args.weights, args.out, clips[fold])
    if args.detect_only:
        return 0

    timing: list[dict] = []          # rows not yet in _timing.csv; save_timing empties it
    failed: dict[str, str] = {}
    for name in args.trackers.split(","):
        if name not in CLASS_MAP:
            print(f"unknown tracker {name}, skipping")
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            for fold in folds:
                run_tracker(name, fold, clips[fold], timing, args.overwrite, args.out, device)
        except Exception as e:                      # one dud must not kill the sweep
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"  !! {name} FAILED -- {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    timing_csv = args.out / "tracks" / "_timing.csv"
    save_timing(timing, timing_csv)                 # last try for anything still pending
    if timing:
        print(f"\n!! {len(timing)} timing row(s) could NOT be written to {timing_csv}. "
              "They follow as CSV so the log keeps them:")
        w = csv.DictWriter(sys.stdout, fieldnames=TIMING_FIELDS)
        w.writeheader()
        w.writerows(timing)
    if failed:
        print("\nFAILED TRACKERS (reported, not hidden):")
        for k, v in failed.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
