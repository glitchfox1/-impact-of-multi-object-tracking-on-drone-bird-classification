#!/usr/bin/env python3
"""Does each tracker honour the shared operating point? Tested by construction, not inferred.

run_trackers.build_tracker CONFIGURES every tracker to start tracks only from detections
>= CONF_HIGH (0.45) and to let detections down to CONF_LOW (0.10) continue existing tracks.
Whether a tracker actually BEHAVES that way cannot be read off its parameters (that is how the
2026-09-24 ConfigError went unnoticed), and it cannot be read cleanly off real output either:
the 09-24 measure, the confidence of a track's first OUTPUT row, is confounded because a track
is emitted only after min_hits / n_init = 3 hits, so its first row usually comes from a later
detection than the one that started it.

So each tracker is fed three synthetic clips (640x512, a dark 30x25 px blob on a textured
grey sky, 30 fps; frames 1-5 empty so no tracker is in its min_hits warm-up):

  S1  sub-threshold only   one object, conf 0.30 throughout          -> rows must be 0
  S2  start high, go low   one object, conf 0.60 for 15 frames, then
                           0.30 for 40                               -> rows in the 0.30 phase
                                                                        = the floor is used
  S3  low newcomer         object A at 0.60 throughout; object B,
                           far from A, appears at frame 26 at 0.30   -> rows for B = a track
                                                                        started below 0.45
                                                                        (boosting trackers do
                                                                        this by design)

--legacy rebuilds the pre-2026-09-25 configuration (no Hybrid-SORT det_thresh override, OC-SORT
use_byte off, no StrongSORT start gate) so the before/after is measured on the same probe.

    python src/paper1/operating_point_probe.py            # current + legacy -> results/operating_point_probe.csv
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_trackers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H, BW, BH = 640, 512, 30, 25
LOW, HIGH = 0.30, 0.60          # below and above CONF_HIGH, both above CONF_LOW


@contextlib.contextmanager
def legacy_config():
    """The operating point as it was before 2026-09-25, for the before/after comparison."""
    saved = (dict(rt.DET_THRESH_OVERRIDE), dict(rt.EXTRA_KWARGS), set(rt.START_GATED))
    rt.DET_THRESH_OVERRIDE.clear(), rt.EXTRA_KWARGS.clear(), rt.START_GATED.clear()
    try:
        yield
    finally:
        rt.DET_THRESH_OVERRIDE.update(saved[0]), rt.EXTRA_KWARGS.update(saved[1])
        rt.START_GATED.update(saved[2])


def sky(seed: int = 0) -> np.ndarray:
    """Textured grey background: ECC camera-motion compensation needs some gradient."""
    rng = np.random.default_rng(seed)
    base = rng.normal(190, 6, (H // 8, W // 8)).clip(0, 255).astype(np.uint8)
    import cv2
    return cv2.cvtColor(cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)


def run(tracker, objects: dict[str, list[tuple[int, float, float, float]]], n_frames: int,
        background: np.ndarray) -> dict[str, list[int]]:
    """objects: name -> [(frame, cx, cy, conf)]. Returns name -> frames in which a track row
    was emitted on that object (matched by IoU >= 0.5 with its box)."""
    import cv2
    by_frame: dict[int, list] = {}
    for name, pts in objects.items():
        for f, cx, cy, c in pts:
            by_frame.setdefault(f, []).append((name, cx, cy, c))
    hits: dict[str, list[int]] = {k: [] for k in objects}
    for f in range(1, n_frames + 1):
        img = background.copy()
        dets = []
        for name, cx, cy, c in by_frame.get(f, []):
            x1, y1 = cx - BW / 2, cy - BH / 2
            cv2.ellipse(img, (int(cx), int(cy)), (BW // 2, BH // 2), 0, 0, 360, (40, 40, 40), -1)
            dets.append([x1, y1, x1 + BW, y1 + BH, c, 0])
        d = np.array(dets, np.float32) if dets else np.zeros((0, 6), np.float32)
        out = tracker.update(d, img)
        out = np.asarray(out if out is not None else np.zeros((0, 8)))
        if out.ndim != 2 or not len(out):
            continue
        for name, cx, cy, c in by_frame.get(f, []):
            gt = np.array([cx - BW / 2, cy - BH / 2, cx + BW / 2, cy + BH / 2])
            ix = np.clip(np.minimum(out[:, 2], gt[2]) - np.maximum(out[:, 0], gt[0]), 0, None)
            iy = np.clip(np.minimum(out[:, 3], gt[3]) - np.maximum(out[:, 1], gt[1]), 0, None)
            inter = ix * iy
            union = (out[:, 2] - out[:, 0]) * (out[:, 3] - out[:, 1]) + BW * BH - inter
            if (inter / union >= 0.5).any():
                hits[name].append(f)
    return hits


def path(f0: int, f1: int, x0: float, y0: float, vx: float, conf) -> list:
    """Horizontal motion at `vx` px per frame; conf is a number or a function of the frame."""
    return [(f, x0 + vx * (f - f0), y0, conf(f) if callable(conf) else conf) for f in range(f0, f1 + 1)]


def probe(name: str, background: np.ndarray) -> dict:
    s1 = run(rt.build_tracker(name, 30.0), {"a": path(6, 65, 100, 200, 3, LOW)}, 65, background)
    s2 = run(rt.build_tracker(name, 30.0),
             {"a": path(6, 60, 100, 200, 3, lambda f: HIGH if f <= 20 else LOW)}, 60, background)
    s3 = run(rt.build_tracker(name, 30.0),
             {"a": path(6, 65, 80, 100, 3, HIGH), "b": path(26, 65, 520, 420, -2, LOW)},
             65, background)
    s2_low = [f for f in s2["a"] if f > 20]
    return {
        "tracker": name,
        "s1_rows_subthreshold_only": len(s1["a"]),
        "s2_rows_high_phase": len([f for f in s2["a"] if f <= 20]),
        "s2_rows_low_phase": len(s2_low),
        "s3_rows_low_newcomer": len(s3["b"]),
        "s3_rows_anchor": len(s3["a"]),
        "start_gate_holds": len(s1["a"]) == 0,
        "floor_used": len(s2_low) > 0,
        "low_newcomer_started": len(s3["b"]) > 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", default=",".join(rt.ALL_TRACKERS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "operating_point_probe.csv")
    ap.add_argument("--current-only", action="store_true", help="skip the legacy configuration")
    args = ap.parse_args()

    background = sky()
    rows = []
    configs = [("current", contextlib.nullcontext)] + ([] if args.current_only
                                                       else [("legacy", legacy_config)])
    for label, ctx in configs:
        for name in args.trackers.split(","):
            with ctx():
                r = {"config": label, **probe(name, background)}
            rows.append(r)
            print(f"{label:8s} {name:11s} S1 {r['s1_rows_subthreshold_only']:3d}  "
                  f"S2 high {r['s2_rows_high_phase']:3d} low {r['s2_rows_low_phase']:3d}  "
                  f"S3 newcomer {r['s3_rows_low_newcomer']:3d} (anchor {r['s3_rows_anchor']:3d})  "
                  f"gate {'OK ' if r['start_gate_holds'] else 'NO '} "
                  f"floor {'used' if r['floor_used'] else 'unused'}", flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
