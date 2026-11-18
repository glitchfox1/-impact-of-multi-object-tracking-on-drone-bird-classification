#!/usr/bin/env python3
"""Ceiling test: can the MSc thesis's 7 track features tell a bird from a drone,
given PERFECT tracks?

No detector, no tracker — features are computed on ground-truth boxes, so this is the
upper bound the full pipeline can aspire to. If the classes do not separate here, no
tracker will rescue them and the pipeline is not worth building.

What it reports
  1. the thesis voting rule (v7frame.py thresholds, >=3 votes = bird) applied as-is
  1b. the same voting rule with thresholds AND directions fitted per training fold
  2. per-feature AUC, i.e. how much each feature alone knows
  3. learned classifiers on the same 7 features, 5-fold grouped CV (folds come from
     the subset manifest, stratified by class x distance bin)
  4. feature-group ablations: drop area / motion / path, and leave-one-out
Every number printed is also written to results/ceiling_summary_<tag>.csv. The paper
quotes these and uses 1b + 3 as the leakage guard, so none may live only in a terminal.

Fixes two defects of the thesis implementation (see DAI fefd34fc):
  * gap-aware differencing — rows are only compared when the frames really are k apart,
    instead of treating consecutive ROWS as consecutive FRAMES
  * speeds are per second (x fps), so clips at different frame rates are comparable
Coordinates are normalised by image size, as in the thesis, so its thresholds still mean
the same thing.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES = ["r2_95", "r3_95", "r5_95", "speed_std", "accel_std", "angle_std", "tortuosity"]
GROUPS = {"area": ["r2_95", "r3_95", "r5_95"],
          "motion": ["speed_std", "accel_std"],
          "path": ["angle_std", "tortuosity"]}
# v7frame.py = the thesis version; a vote is cast when the feature exceeds its threshold
THESIS_THR = {"r2_95": 1.05, "r3_95": 1.10, "r5_95": 1.20,
              "speed_std": 0.004, "accel_std": 0.005, "angle_std": 0.7, "tortuosity": 1.3}
THESIS_VOTES = 3
# The thesis set its speed and acceleration thresholds on PER-FRAME speeds; features_for
# reports both PER SECOND (x fps). They are converted back before the published thresholds
# are applied -- without that, at 30 fps speed_std voted bird on 95.7% of ceiling tracks and
# the published rule was scored at 0.231 instead of 0.375 (docs/errors.md, 2026-09-21).
THESIS_PER_FRAME = ("speed_std", "accel_std")
MIN_FRAMES = 6


def read_seq(seq_dir: Path):
    """Ground-truth track -> (frames, cx, cy, area) in normalised units."""
    ini = dict(line.split("=", 1) for line in (seq_dir / "seqinfo.ini").read_text().splitlines()
               if "=" in line)
    W, H, fps = int(ini["imWidth"]), int(ini["imHeight"]), float(ini["frameRate"])
    rows = []
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        f, _tid, x, y, w, h = line.split(",")[:6]
        x, y, w, h = float(x), float(y), float(w), float(h)
        rows.append((int(f), (x + w / 2) / W, (y + h / 2) / H, (w / W) * (h / H)))
    if not rows:                   # every label dropped by the ADR-006 corrections (V_BIRD_020)
        e = np.array([], dtype=float)
        return e.astype(int), e, e, e, fps
    rows.sort()
    a = np.array(rows, dtype=float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], fps


def roll_med(v: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - k // 2): i + k // 2 + 1])
    return out


def ratio95(frames: np.ndarray, area: np.ndarray, lag: int) -> float:
    """95th percentile of area(t+lag)/area(t), using only pairs really `lag` apart."""
    idx = {f: i for i, f in enumerate(frames)}
    r = [area[idx[f + lag]] / area[i] for i, f in enumerate(frames)
         if f + lag in idx and area[i] > 0]
    return float(np.percentile(r, 95)) if r else 1.0


def features_for(frames, cx, cy, area, fps) -> dict | None:
    if len(frames) < MIN_FRAMES:
        return None
    cx, cy, area = roll_med(cx), roll_med(cy), roll_med(area)
    step = np.diff(frames)
    adj = step == 1                                   # gap-aware: neighbours only
    dx, dy = np.diff(cx)[adj], np.diff(cy)[adj]
    v = np.hypot(dx, dy) * fps                        # per second, not per frame
    if len(v) < 3:
        return None
    ang = []
    for i in range(len(dx) - 1):
        n1, n2 = np.hypot(dx[i], dy[i]), np.hypot(dx[i + 1], dy[i + 1])
        if n1 > 1e-9 and n2 > 1e-9:
            c = (dx[i] * dx[i + 1] + dy[i] * dy[i + 1]) / (n1 * n2)
            ang.append(np.arccos(np.clip(c, -1, 1)))
    # Tortuosity's path must span the same frames as its displacement. Summing only the
    # adjacent-frame steps (dx, dy above) drops every gap, so a gappy track could score
    # below 1 -- impossible for a real path (5 of 33 bytetrack bird tracks, 2026-09-21).
    # Across a gap the straight chord is used: the shortest path consistent with the
    # boxes, so tortuosity is always >= 1. Speed and turning stay adjacent-only.
    path = float(np.hypot(np.diff(cx), np.diff(cy)).sum())
    disp = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
    duration = (frames[-1] - frames[0]) / fps
    return {
        "r2_95": ratio95(frames, area, 1),
        "r3_95": ratio95(frames, area, 2),
        "r5_95": ratio95(frames, area, 4),
        "speed_std": float(np.std(v)),
        "accel_std": float(np.std(np.diff(v))),
        "angle_std": float(np.nanstd(ang)) if ang else 0.0,
        "tortuosity": min(path / disp, 10.0) if disp > 1e-9 else 10.0,
        # Diagnostics, NEVER classifier inputs (every classifier selects FEATURES):
        # n_frames measures how long the tracker held on; net_disp_s (net displacement
        # per second) tests whether tortuosity is just a hovering detector.
        "n_frames": len(frames),
        "net_disp_s": disp / duration if duration > 0 else 0.0,
    }


def thesis_vote(row, fps: float) -> str:
    """The thesis rule as published, in the units its thresholds were set in."""
    votes = sum((row[f] / fps if f in THESIS_PER_FRAME else row[f]) > THESIS_THR[f]
                for f in FEATURES)
    return "BIRD" if votes >= THESIS_VOTES else "DRONE"


def fit_vote_rule(X: pd.DataFrame, y: np.ndarray):
    """Learn the voting rule from training data: per-feature cut-off AND direction,
    then the number of votes needed.

    The thesis fixed both by hand on 10 bird videos against 2 drone videos, so its
    thresholds are biased toward bird (they call 80% of everything bird) and five of
    the seven features point the wrong way on other footage. Direction is therefore
    learned, not assumed.
    """
    # Iterate the columns ACTUALLY passed, not the global FEATURES: feature_selection.py
    # fits this rule on subsets of one or more features. Identical behaviour when all 7
    # are passed, which is every other caller.
    rules = {}
    for f in X.columns:
        vals = np.unique(np.percentile(X[f], np.arange(5, 100, 5)))
        best = None
        for t in vals:
            for sign in (1, -1):
                pred = np.where((X[f] > t) if sign == 1 else (X[f] < t), "BIRD", "DRONE")
                s = f1_score(y, pred, average="macro")
                if best is None or s > best[0]:
                    best = (s, float(t), sign)
        rules[f] = (best[1], best[2])
    votes_tr = np.array([sum((X.iloc[i][f] > t) if s == 1 else (X.iloc[i][f] < t)
                             for f, (t, s) in rules.items()) for i in range(len(X))])
    best_v, best_s = 3, -1
    for v in range(1, len(rules) + 1):
        s = f1_score(y, np.where(votes_tr >= v, "BIRD", "DRONE"), average="macro")
        if s > best_s:
            best_v, best_s = v, s
    return rules, best_v


def apply_vote_rule(X: pd.DataFrame, rules: dict, min_votes: int) -> np.ndarray:
    votes = np.array([sum((X.iloc[i][f] > t) if s == 1 else (X.iloc[i][f] < t)
                          for f, (t, s) in rules.items()) for i in range(len(X))])
    return np.where(votes >= min_votes, "BIRD", "DRONE")


def cv_vote_rule(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, cols=None):
    cols = cols or FEATURES
    pred = np.empty(len(y), dtype=object)
    picked = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        rules, v = fit_vote_rule(X.loc[tr, cols], y[tr])
        pred[te] = apply_vote_rule(X.loc[te, cols], rules, v)
        picked.append((rules, v))
    return f1_score(y, pred, average="macro"), (pred == y).mean(), picked


def cv_macro_f1(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, model_fn) -> float:
    pred = [None] * len(y)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = model_fn()
        m.fit(X[tr], y[tr])
        for i, p in zip(np.where(te)[0], m.predict(X[te])):
            pred[i] = p                      # keep y's dtype: str labels or 0/1 for xgboost
    return f1_score(y, np.asarray(pred), average="macro")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=root / "gt" / "svanstrom_visible")
    ap.add_argument("--classes", nargs="+", default=["BIRD", "DRONE"])
    ap.add_argument("--window", type=int, default=0,
                    help="split each track into N-frame windows (0 = whole track)")
    ap.add_argument("--out", type=Path, default=root / "results")
    args = ap.parse_args()

    man = pd.read_csv(args.gt / "_manifest.csv")
    man = man[(man.selected == 1) & (man["class"].isin(args.classes))]

    rows, fps_of = [], {}
    for _, r in man.iterrows():
        frames, cx, cy, area, fps = read_seq(args.gt / r.video)
        fps_of[r.video] = fps
        chunks = ([(frames, cx, cy, area)] if args.window <= 0 else
                  [(frames[i:i + args.window], cx[i:i + args.window],
                    cy[i:i + args.window], area[i:i + args.window])
                   for i in range(0, len(frames), args.window)])
        for j, (f, x, y, a) in enumerate(chunks):
            feat = features_for(f, x, y, a, fps)
            if feat:
                rows.append({"video": r.video, "chunk": j, "cls": r["class"],
                             "fold": r.fold, "bin": r.distance_bin, **feat})
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "track" if args.window <= 0 else f"win{args.window}"
    df.to_csv(args.out / f"ceiling_features_{tag}.csv", index=False)

    y = df.cls.values
    X = df[FEATURES]
    folds = df.fold.values
    print(f"samples: {len(df)} ({dict(df.cls.value_counts())}), unit = "
          f"{'whole track' if args.window <= 0 else str(args.window) + '-frame window'}\n")

    pred = np.array([thesis_vote(row, fps_of[row.video]) for _, row in df.iterrows()])
    acc = (pred == y).mean()
    print("1) THESIS VOTING RULE, applied as-is")
    print(f"   accuracy {acc:.1%} | macro F1 {f1_score(y, pred, average='macro'):.3f} | "
          f"predicted bird: {(pred == 'BIRD').mean():.0%}")
    for cls in args.classes:
        m = y == cls
        print(f"   {cls:6s} n={m.sum():3d}  correct {(pred[m] == cls).mean():.1%}")

    f1v, accv, picked = cv_vote_rule(X, y, folds)
    print("\n1b) SAME VOTING RULE, thresholds + directions FITTED per training fold")
    print(f"   accuracy {accv:.1%} | macro F1 {f1v:.3f}")
    rules0, v0 = picked[0]
    rule0_text = f"votes>={v0}, " + ", ".join(
        f"{f}{'>' if s == 1 else '<'}{t:.3f}" for f, (t, s) in rules0.items())
    print(f"   fold 0 learned: {rule0_text}")
    flipped = [f for f, (_t, s) in rules0.items() if s == -1]
    print(f"   features the data flipped vs the thesis: {flipped or 'none'}")

    print("\n2) PER-FEATURE AUC (1.0 or 0.0 = perfectly separating on its own)")
    ybin = (y == "BIRD").astype(int)
    aucs = {f: roc_auc_score(ybin, df[f]) for f in FEATURES}
    for f, a in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        print(f"   {f:12s} {a:.3f}   thesis threshold {THESIS_THR[f]:<6} "
              f"median bird {df[ybin == 1][f].median():.4f} vs drone {df[ybin == 0][f].median():.4f}")

    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = lambda: make_pipeline(
            StandardScaler(), XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                            verbosity=0, use_label_encoder=False,
                                            eval_metric="logloss"))
    except Exception:
        pass

    print("\n3) LEARNED CLASSIFIERS on the same 7 features (5-fold CV, macro F1)")
    base = {}
    for name, fn in models.items():
        if name == "xgboost":
            yy = (y == "BIRD").astype(int)
            base[name] = cv_macro_f1(X, yy, folds, fn)
        else:
            base[name] = cv_macro_f1(X, y, folds, fn)
        print(f"   {name:14s} {base[name]:.3f}")

    print("\n4) ABLATIONS (macro F1, random forest, 5-fold CV)")
    full = cv_macro_f1(X, y, folds, models["random forest"])
    print(f"   all 7 features        {full:.3f}")
    ablation = {"ablation_rf_all7": full}
    for g, cols in GROUPS.items():
        keep = [f for f in FEATURES if f not in cols]
        s = cv_macro_f1(df[keep], y, folds, models["random forest"])
        ablation[f"ablation_rf_minus_{g}"] = s
        print(f"   minus {g:-<16s} {s:.3f}   ({s - full:+.3f})")
    print("   leave-one-out:")
    for f in FEATURES:
        keep = [c for c in FEATURES if c != f]
        s = cv_macro_f1(df[keep], y, folds, models["random forest"])
        ablation[f"ablation_rf_without_{f}"] = s
        print(f"     without {f:12s} {s:.3f}   ({s - full:+.3f})")

    with (args.out / f"ceiling_summary_{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["samples", len(df)])
        w.writerow(["thesis_rule_accuracy", f"{acc:.4f}"])
        w.writerow(["thesis_rule_macro_f1", f"{f1_score(y, pred, average='macro'):.4f}"])
        w.writerow(["vote_rule_fitted_accuracy", f"{accv:.4f}"])
        w.writerow(["vote_rule_fitted_macro_f1", f"{f1v:.4f}"])
        # Fold 0's learned rule, kept numeric so every value in this file parses as a
        # number. sign +1 = the thesis's direction (">"), -1 = flipped ("<").
        w.writerow(["vote_rule_fold0_min_votes", v0])
        w.writerow(["vote_rule_fold0_n_flipped", len(flipped)])
        for f, (t, s) in rules0.items():
            w.writerow([f"vote_rule_fold0_sign_{f}", s])
            w.writerow([f"vote_rule_fold0_thr_{f}", f"{t:.4f}"])
        for f, a in aucs.items():
            w.writerow([f"auc_{f}", f"{a:.4f}"])
        for name, s in base.items():
            w.writerow([f"cv_macro_f1_{name}", f"{s:.4f}"])
        # Scores, not deltas: the delta is this minus ablation_rf_all7.
        for name, s in ablation.items():
            w.writerow([name, f"{s:.4f}"])
    print(f"\n-> {args.out}/ceiling_features_{tag}.csv, ceiling_summary_{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
