#!/usr/bin/env python3
"""Setup and cost tables for the Paper 1 manuscript, generated from the code that ran.

--settings   every tracker is built exactly as run_trackers.build_tracker built it, and the
             values are read back from the INSTANCE, not typed from documentation. max_age,
             min_hits and iou_threshold arrive through BoxMOT's base-class kwargs, so a
             signature alone would not show them.
             -> results/tracker_settings.csv

--benchmark  1. detector: parameters and GFLOPs of all five fold detectors (must be equal);
                latency of the fold-0 detector on its held-out frames, FP32, 640 px, 300-frame warm-up,
                batch 1 (per-frame deployment) and batch 16 (the caching run), after a
                warm-up, repeated --repeats times so the spread is visible
             2. trackers: association speed from the fold-0 rows of results/tracks/_timing.csv
                only (ADR-001: one process, all nine trackers, 32 clips). Excludes tracker
                construction; the ReID device of each tracker is read from the timing file
                (CPU until 2026-09-25, GPU from the 2026-09-26 rerun).
             3. features + random forest: per-track time on the perfect-track reference
             -> results/cost_benchmark.csv

Run the benchmark on an otherwise idle machine: pre- and post-processing run on the CPU.

    python src/paper1/paper_tables.py --settings
    python src/paper1/paper_tables.py --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_trackers import (  # noqa: E402
    ALL_TRACKERS, CONF_HIGH, CONF_LOW, HIGH_ALIASES, LOW_ALIASES, START_GATED, build_tracker,
    fold_videos, frames_of,
)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
REID = {"botsort", "strongsort", "deepocsort", "hybridsort", "occluboost"}


# ------------------------------------------------------------------ settings

def reid_device(name: str) -> str:
    """Where the tracker's ReID network actually ran, from the per-clip device column of
    results/tracks/_timing.csv (added 2026-09-26; older files have none: BoxMOT's default, CPU)."""
    t = pd.read_csv(RES / "tracks" / "_timing.csv")
    if "device" not in t:
        return "CPU"
    d = {x for x in t.loc[t.tracker == name, "device"].astype(str) if not x.startswith("n/a")}
    return "GPU" if d == {"0"} else "CPU" if d == {"cpu"} else "mixed CPU/GPU" if d else "n/a"


def measured_confidence(name: str) -> dict:
    """The lowest confidence in any written output row (column 7) -- shows whether low
    detections reach the output. BoostTrack rewrites the confidence of detections it boosts
    to the threshold, so its low-confidence use cannot be seen here.

    The 2026-09-24 version also reported the share of tracks whose FIRST OUTPUT ROW is below
    0.45 as "tracks started below the start threshold". That is confounded: a track is emitted
    only after min_hits / n_init = 3 hits, so its first row usually comes from a later, often
    low-confidence, detection (after the 2026-09-25 fix StrongSORT still showed 43-100% on
    bird clips while it provably never starts below 0.45). Whether a tracker starts below
    0.45 now comes from operating_point_probe.py, which tests it by construction."""
    confs, n_tracks = [], 0
    for f in sorted((RES / "tracks" / name).glob("*.txt")):
        if not f.stat().st_size:
            continue
        a = np.loadtxt(f, delimiter=",", ndmin=2)
        confs.append(a[:, 6])
        n_tracks += len(np.unique(a[:, 1]))
    return {"measured_tracks": n_tracks,
            "measured_lowest_conf_used": round(float(np.concatenate(confs).min()), 3)}


def probe_verdicts() -> pd.DataFrame:
    """operating_point_probe.py results for the configuration in use."""
    p = RES / "operating_point_probe.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing: run python src/paper1/operating_point_probe.py first")
    df = pd.read_csv(p)
    return df[df.config == "current"].set_index("tracker")


def regime(r: dict) -> str:
    if not r["probe_start_gate_holds"]:
        return f"starts tracks below {CONF_HIGH}"
    if not r["probe_floor_used"]:
        return f"only detections >= {CONF_HIGH}; no low-confidence pass"
    if r["probe_low_newcomer_started"]:
        return (f"start {CONF_HIGH}, floor {CONF_LOW}; confidence boosting also lets a low "
                "detection far from all tracks start one (by design)")
    return f"start {CONF_HIGH}, floor {CONF_LOW}, as configured"


def settings() -> pd.DataFrame:
    """Configured values read from the built instance, beside measured behaviour from the tracks.
    The shared operating point is what run_trackers CONFIGURED; the measured columns show which
    trackers honour it (2026-09-24: only ByteTrack, SFSORT and BoT-SORT did; fixed 2026-09-25,
    see run_trackers DET_THRESH_OVERRIDE / EXTRA_KWARGS / START_GATED)."""
    probe = probe_verdicts()
    rows = []
    for name in ALL_TRACKERS:
        t = build_tracker(name, 30.0)
        v = vars(t)
        high = [a for a in HIGH_ALIASES if a in v]
        low = [a for a in LOW_ALIASES if a in v]
        rows.append({
            "tracker": name,
            "start_param": ("start gate in code (run_trackers.gate_track_starts)"
                            if name in START_GATED else "+".join(high) or "det_thresh"),
            "start": v[high[0]] if high else v["det_thresh"],
            "use_byte": v.get("use_byte", ""),
            "floor_param": "+".join(low) or "det_thresh",
            "floor": v[low[0]] if low else v["det_thresh"],
            "det_thresh": v["det_thresh"],
            "max_age": v["max_age"],
            "track_buffer": v.get("track_buffer", ""),
            "min_hits": v["min_hits"],
            "iou_threshold": v["iou_threshold"],
            "appearance": (f"OSNet x0.25 ({reid_device(name)})"
                           if v.get("use_embeddings", name == "strongsort") else "none"),
            # the instance's `cmc` object is what runs (ECC, or SOF = sparse optical flow)
            "camera_motion_comp": (type(v["cmc"]).__name__ if v.get("cmc") is not None
                                   and not v.get("cmc_off", False) else "none"),
            **measured_confidence(name),
            **{f"probe_{c}": bool(probe.loc[name, c])
               for c in ("start_gate_holds", "floor_used", "low_newcomer_started")},
        })
        rows[-1]["effective_regime"] = regime(rows[-1])
    df = pd.DataFrame(rows)
    if (df.start != CONF_HIGH).any():
        raise SystemExit(f"start threshold not configured as {CONF_HIGH}:\n{df}")
    return df


# ------------------------------------------------------------------ benchmark

def detector_rows(repeats: int, n_frames: int) -> list[dict]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    rows = []
    infos = []
    for k in range(5):
        m = YOLO(str(ROOT / "runs" / "detect" / f"fold{k}" / "weights" / "best.pt"))
        layers, params, _grads, gflops = m.info()     # ultralytics 8.4 returns None when verbose=False
        infos.append((layers, params, round(gflops, 2)))
    if len(set(infos)) != 1:
        raise SystemExit(f"fold detectors differ in size: {infos}")
    layers, params, gflops = infos[0]
    env = (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
           f"CUDA {torch.version.cuda}, ultralytics {ultralytics.__version__}")
    rows += [{"component": "detector", "metric": "parameters", "value": params, "unit": "count", "note": "YOLOv8s, 1 class; identical for the 5 folds"},
             {"component": "detector", "metric": "GFLOPs", "value": gflops, "unit": "GFLOPs", "note": "at 640 px"},
             {"component": "detector", "metric": "layers", "value": layers, "unit": "count", "note": ""}]

    model = YOLO(str(ROOT / "runs" / "detect" / "fold0" / "weights" / "best.pt"))
    frames = [str(p) for v in fold_videos(0) for p in frames_of(v)]
    warm, frames = frames[:300], frames[300:300 + n_frames]   # laptop GPU: 50 frames left the clocks unsettled (24% spread)
    for p in warm:
        model.predict(p, imgsz=640, conf=0.10, device=0, verbose=False)
    for rep in range(repeats):
        pre, inf, post = [], [], []
        for p in frames:
            r = model.predict(p, imgsz=640, conf=0.10, device=0, verbose=False)[0]
            pre.append(r.speed["preprocess"]), inf.append(r.speed["inference"]), post.append(r.speed["postprocess"])
        tot = np.array(pre) + np.array(inf) + np.array(post)
        tag = f"batch 1, run {rep + 1}/{repeats}, {len(frames)} frames, FP32; {env}"
        rows += [{"component": "detector", "metric": f"{s}_ms_median", "value": round(float(np.median(a)), 2),
                  "unit": "ms/frame", "note": tag} for s, a in (("preprocess", pre), ("inference", inf),
                                                                 ("postprocess", post), ("total", tot))]
        rows.append({"component": "detector", "metric": "fps_batch1", "value": round(1000 / float(np.median(tot)), 1),
                     "unit": "frames/s", "note": tag})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(0, len(frames), 16):
            model.predict(frames[s:s + 16], imgsz=640, conf=0.10, device=0, verbose=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"component": "detector", "metric": "fps_batch16", "value": round(len(frames) / dt, 1),
                     "unit": "frames/s", "note": f"batch 16 wall clock incl. JPEG decode, run {rep + 1}/{repeats}"})
    return rows


def tracker_rows() -> list[dict]:
    t = pd.read_csv(RES / "tracks" / "_timing.csv")
    t = t[t.fold == 0]
    rows = []
    for name, g in t.groupby("tracker"):
        fps = g.frames.sum() / g.seconds.sum()
        rows.append({"component": f"tracker:{name}", "metric": "fps_association", "value": round(fps, 1),
                     "unit": "frames/s", "note": f"fold 0, {len(g)} clips, {int(g.frames.sum())} frames; "
                     f"association only, construction excluded"
                     f"{f'; ReID on {reid_device(name)}' if name in REID else ''}"})
    return rows


def classifier_rows() -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from feature_ceiling import FEATURES, features_for, read_seq
    man = pd.read_csv(ROOT / "gt" / "svanstrom_visible" / "_manifest.csv")
    man = man[(man.selected == 1) & man["class"].isin(["BIRD", "DRONE"])]
    seqs = [read_seq(ROOT / "gt" / "svanstrom_visible" / v) for v in man.video]
    t0 = time.perf_counter()
    feats = [features_for(f, x, y, a, fps) for f, x, y, a, fps in seqs]
    t_feat = (time.perf_counter() - t0) / len(seqs)
    keep = [i for i, f in enumerate(feats) if f]
    X = pd.DataFrame([feats[i] for i in keep])[FEATURES]
    y = man["class"].values[keep]
    rf = RandomForestClassifier(n_estimators=300, random_state=0)
    t0 = time.perf_counter()
    rf.fit(X, y)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(len(X)):
        rf.predict(X.iloc[[i]])
    t_pred = (time.perf_counter() - t0) / len(X)
    n = len(X)
    return [{"component": "features", "metric": "ms_per_track", "value": round(1000 * t_feat, 2), "unit": "ms/track",
             "note": f"7 features, averaged over {len(seqs)} clips' GT tracks ({n} long enough to score)"},
            {"component": "random_forest", "metric": "fit_s", "value": round(t_fit, 2), "unit": "s", "note": f"300 trees, {n} tracks"},
            {"component": "random_forest", "metric": "ms_per_track_predict", "value": round(1000 * t_pred, 2),
             "unit": "ms/track", "note": "one track at a time"}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--frames", type=int, default=2000)
    args = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 20)
    if args.settings:
        s = settings()
        s.to_csv(RES / "tracker_settings.csv", index=False)
        print(s.to_string(index=False), f"\n-> {RES / 'tracker_settings.csv'}")
    if args.benchmark:
        b = pd.DataFrame(detector_rows(args.repeats, args.frames) + tracker_rows() + classifier_rows())
        b.to_csv(RES / "cost_benchmark.csv", index=False)
        print(b.to_string(index=False), f"\n-> {RES / 'cost_benchmark.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
