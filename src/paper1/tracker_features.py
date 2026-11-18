#!/usr/bin/env python3
"""Stage 3: what are the 7 track features worth on REAL tracks?

feature_ceiling.py answered the question on perfect ground-truth tracks and got
macro F1 0.771 (fitted voting) / 0.871 (random forest). Those are upper bounds: they
assume a detector that never misses and a tracker that never breaks a track. This
script re-asks the identical question on tracks produced by a trained detector plus
each boxmot tracker, so the gap between the two is exactly the cost of the perception
stack.

Feature code, classifier suite and fold assignment are IMPORTED from feature_ceiling
rather than re-implemented, so the only thing that changes between the two tables is
where the boxes came from.

Two track-selection policies are reported, because they answer different questions:
  oracle   the tracker track overlapping ground truth most -- best case, isolates
           feature quality from selection error
  longest  the longest track in the clip, chosen without looking at ground truth --
           what a deployed system can actually do
The oracle row is NOT a deployable result and is never reported alone.

Writes results/tracker_features_<tracker>.csv, results/tracker_quality.csv and
results/tracker_classification.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_ceiling import (  # noqa: E402
    FEATURES, GROUPS, MIN_FRAMES, cv_macro_f1, cv_vote_rule, features_for, thesis_vote,
)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
GT_DIR = ROOT / "gt" / "svanstrom_visible"


# --------------------------------------------------------------------------- io

def read_mot(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """MOT file -> {frame: [(track_id, x, y, w, h), ...]}."""
    out: dict[int, list] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        out[int(p[0])].append((int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return out


def seq_meta(video: str) -> tuple[int, int, float]:
    ini = dict(l.split("=", 1) for l in (GT_DIR / video / "seqinfo.ini").read_text().splitlines()
               if "=" in l)
    return int(ini["imWidth"]), int(ini["imHeight"]), float(ini["frameRate"])


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------- matching

def match_clip(video: str, tk: dict, thr: float) -> dict:
    """Align tracker output to ground truth for one single-target clip.

    Returns the per-clip tracking-quality row plus, for each selection policy, the
    track id to hand to the feature extractor.
    """
    gt = read_mot(GT_DIR / video / "gt" / "gt.txt")
    gt_frames = sorted(gt)

    hits_by_id: dict[int, int] = defaultdict(int)     # track id -> frames matched to GT
    matched_seq: list[tuple[int, int]] = []           # (frame, track id) where matched
    n_matched = 0
    for f in gt_frames:
        g = gt[f][0][1:]                              # single target per frame
        best_id, best_iou = None, thr
        for tid, x, y, w, h in tk.get(f, []):
            v = iou(g, (x, y, w, h))
            if v >= best_iou:
                best_id, best_iou = tid, v
        if best_id is not None:
            hits_by_id[best_id] += 1
            matched_seq.append((f, best_id))
            n_matched += 1

    # ID switches: the GT object changing hands between consecutive MATCHED frames.
    id_sw = sum(1 for i in range(1, len(matched_seq))
                if matched_seq[i][1] != matched_seq[i - 1][1])

    rows_by_id: dict[int, int] = defaultdict(int)
    for f, dets in tk.items():
        for tid, *_ in dets:
            rows_by_id[tid] += 1

    oracle_id = max(hits_by_id, key=hits_by_id.get) if hits_by_id else None
    longest_id = max(rows_by_id, key=rows_by_id.get) if rows_by_id else None

    cover = hits_by_id[oracle_id] / len(gt_frames) if oracle_id is not None else 0.0
    purity = hits_by_id[oracle_id] / rows_by_id[oracle_id] if oracle_id is not None else 0.0
    return {
        "video": video,
        "gt_frames": len(gt_frames),
        "det_recall": n_matched / len(gt_frames) if gt_frames else 0.0,
        "n_tracks": len(rows_by_id),
        "n_tracks_on_gt": len(hits_by_id),
        "id_switches": id_sw,
        "oracle_id": oracle_id,
        "longest_id": longest_id,
        "oracle_coverage": cover,
        "oracle_purity": purity,
        "longest_is_oracle": int(oracle_id is not None and oracle_id == longest_id),
    }


def track_series(tk: dict, tid: int, W: int, H: int):
    """One track id -> (frames, cx, cy, area) normalised, as feature_ceiling expects."""
    rows = []
    for f, dets in tk.items():
        for t, x, y, w, h in dets:
            if t == tid:
                rows.append((f, (x + w / 2) / W, (y + h / 2) / H, (w / W) * (h / H)))
    rows.sort()
    if not rows:
        return None
    a = np.array(rows, dtype=float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3]


# ----------------------------------------------------------------- evaluation

def evaluate(df: pd.DataFrame, label: str, out_rows: list) -> None:
    """Run the identical classifier suite feature_ceiling uses, on tracker features."""
    if len(df) < 20 or df.cls.nunique() < 2:
        print(f"   {label}: only {len(df)} usable tracks "
              f"({dict(df.cls.value_counts())}) -- not scored")
        out_rows.append({"unit": label, "n": len(df), "note": "too few usable tracks"})
        return
    y, X, folds = df.cls.values, df[FEATURES], df.fold.values
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=0)  # noqa: E731

    thesis = np.array([thesis_vote(r, seq_meta(r.video)[2]) for _, r in df.iterrows()])
    f1_thesis = f1_score(y, thesis, average="macro")
    f1_vote, acc_vote, _ = cv_vote_rule(X, y, folds)
    f1_rf = cv_macro_f1(X, y, folds, rf)

    row = {"unit": label, "n": len(df),
           "n_bird": int((y == "BIRD").sum()), "n_drone": int((y == "DRONE").sum()),
           "thesis_macro_f1": round(f1_thesis, 4),
           "fitted_vote_macro_f1": round(f1_vote, 4),
           "rf_macro_f1": round(f1_rf, 4)}
    for g, cols in GROUPS.items():
        keep = [f for f in FEATURES if f not in cols]
        row[f"rf_minus_{g}"] = round(cv_macro_f1(df[keep], y, folds, rf) - f1_rf, 4)
    out_rows.append(row)

    print(f"   {label:28s} n={len(df):3d} "
          f"(bird {row['n_bird']}, drone {row['n_drone']})  "
          f"thesis {f1_thesis:.3f} | fitted vote {f1_vote:.3f} | RF {f1_rf:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", default="")
    ap.add_argument("--classes", nargs="+", default=["BIRD", "DRONE"])
    ap.add_argument("--iou", type=float, default=0.3,
                    help="GT/track association IoU; targets are ~30px so 0.5 is harsh")
    args = ap.parse_args()

    man = pd.read_csv(GT_DIR / "_manifest.csv")
    man = man[(man.selected == 1) & (man["class"].isin(args.classes))]

    track_root = RES / "tracks"
    names = ([t for t in args.trackers.split(",") if t] or
             sorted(p.name for p in track_root.iterdir() if p.is_dir()))
    if not names:
        print("no tracker output under results/tracks -- run run_trackers.py first")
        return 1

    quality, classif = [], []
    for name in names:
        print(f"\n=== {name} ===")
        feats, qrows = [], []
        for _, r in man.iterrows():
            f = track_root / name / f"{r.video}.txt"
            if not f.exists():
                continue                      # clip belongs to a fold not yet tracked
            W, H, fps = seq_meta(r.video)
            tk = read_mot(f)
            q = match_clip(r.video, tk, args.iou)
            q.update(tracker=name, cls=r["class"], fold=r.fold, bin=r.distance_bin)
            # A clip whose verified GT is shorter than a usable track (ADR-006: V_BIRD_020 has
            # none left, V_BIRD_021 four frames) has nothing to measure tracking quality or
            # pick an oracle track against. It stays in the longest policy, which needs only
            # the clip's class -- scoring it here would count a correct track as a miss.
            gt_ok = q["gt_frames"] >= MIN_FRAMES
            if gt_ok:
                qrows.append(q)
            for policy, tid in (("oracle", q["oracle_id"] if gt_ok else None),
                                ("longest", q["longest_id"])):
                if tid is None:
                    continue
                s = track_series(tk, tid, W, H)
                if s is None:
                    continue
                feat = features_for(*s, fps)
                if feat:
                    feats.append({"video": r.video, "cls": r["class"], "fold": r.fold,
                                  "bin": r.distance_bin, "policy": policy,
                                  "tracker": name, **feat})
        if not qrows:
            print("   no clips tracked yet, skipping")
            continue

        q = pd.DataFrame(qrows)
        quality.append(q)
        print(f"   clips {len(q)} | det recall {q.det_recall.mean():.3f} | "
              f"tracks/clip {q.n_tracks.mean():.1f} | fragments on GT "
              f"{q.n_tracks_on_gt.mean():.2f} | ID switches {q.id_switches.sum()} | "
              f"oracle coverage {q.oracle_coverage.mean():.3f} | "
              f"longest==oracle {q.longest_is_oracle.mean():.0%}")

        fdf = pd.DataFrame(feats)
        if len(fdf):
            fdf.to_csv(RES / f"tracker_features_{name}.csv", index=False)
            for policy in ("oracle", "longest"):
                sub = fdf[fdf.policy == policy].reset_index(drop=True)
                rows: list = []
                evaluate(sub, f"{name}/{policy}", rows)
                for row in rows:
                    row["tracker"] = name
                    row["policy"] = policy
                classif += rows

    if quality:
        pd.concat(quality).to_csv(RES / "tracker_quality.csv", index=False)
        print(f"\nwrote {RES / 'tracker_quality.csv'}")
    if classif:
        pd.DataFrame(classif).to_csv(RES / "tracker_classification.csv", index=False)
        print(f"wrote {RES / 'tracker_classification.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
