#!/usr/bin/env python3
"""Aruna's ablation: start from NO features, add them one at a time, find the best combination.

feature_ceiling.py and tracker_features.py answer "what do all 7 features give, and what does
removing one cost?" -- a SUBTRACTIVE ablation. This script asks the opposite question:

    baseline with no features at all
    -> each feature ALONE (7 runs)
    -> the best pair, the best triple, ... up to all 7
    -> the best combination overall

With 7 features there are only 127 non-empty subsets, so nothing is guessed: every subset is
scored. The greedy "add the next best feature" path is reported too, because that is the story
Aruna asked for, but the exhaustive winner is what should be quoted -- greedy can miss a pair
that only works together.

Both decision rules are scored on every subset:
  * the fitted voting rule  -- Aruna's method. With k features it learns each one's threshold
    AND direction on the training folds, then needs >= v votes. With k=1 it is a single threshold.
  * a random forest         -- the upper bound of what the subset can support.

HONESTY CHECK (--nested). Picking the best of 127 subsets by cross-validated score and then
quoting that score is optimistic: the selection has seen every fold. --nested repeats the whole
selection INSIDE each outer fold and scores the choice on data the selection never saw. Expect
the nested number to be lower; the gap is how much of the winner's margin was selection luck.

    python src/paper1/feature_selection.py                          # perfect GT tracks
    python src/paper1/feature_selection.py --tracker bytetrack --policy longest
    python src/paper1/feature_selection.py --nested                 # + the honesty check

Writes results/feature_selection_<unit>.csv
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_ceiling import (  # noqa: E402
    FEATURES, apply_vote_rule, cv_macro_f1, cv_vote_rule, fit_vote_rule,
)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def rf():
    return RandomForestClassifier(n_estimators=300, random_state=0)


# ------------------------------------------------------------------ baselines

def cv_majority(y: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    """No features at all: predict the training folds' majority class.

    This is the honest "before any feature" floor. On a balanced 35/35 set it scores 50%
    accuracy and macro F1 0.333 -- one class is always right, the other always wrong.
    """
    pred = np.empty(len(y), dtype=object)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        vals, counts = np.unique(y[tr], return_counts=True)
        pred[te] = vals[counts.argmax()]
    return f1_score(y, pred, average="macro"), float((pred == y).mean())


# ------------------------------------------------------------- subset scoring

def rf_cv_predict(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                  cols: list[str]) -> np.ndarray:
    """Out-of-fold random forest predictions — ONE fit per fold.

    Macro F1 and accuracy are both read off these same predictions. The earlier version
    fitted the forest twice per fold, once inside cv_macro_f1 and once again for the
    accuracy loop: same folds, same columns, same random_state, therefore the same trees
    and the same predictions. Dropping the second fit is arithmetically identical and
    halves the run.
    """
    pred = np.empty(len(y), dtype=object)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = rf().fit(X.loc[tr, cols], y[tr])
        pred[te] = m.predict(X.loc[te, cols])
    return pred


def score_subset(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                 cols: tuple[str, ...]) -> dict:
    f1_vote, acc_vote, _ = cv_vote_rule(X, y, folds, list(cols))
    pred = rf_cv_predict(X, y, folds, list(cols))
    f1_rf = f1_score(y, pred, average="macro")
    acc_rf = float((pred == y).mean())

    return {"k": len(cols), "features": "+".join(cols),
            "vote_macro_f1": round(f1_vote, 4), "vote_accuracy": round(acc_vote, 4),
            "rf_macro_f1": round(f1_rf, 4), "rf_accuracy": round(acc_rf, 4)}


def all_subsets(X, y, folds, jobs: int = 1) -> pd.DataFrame:
    """Score every subset. Subsets are independent, so they fan out over processes.

    Parallelism is orchestration only — each subset is scored by the same code on the
    same folds, so the table does not depend on `jobs`. Keep the forest itself
    single-threaded (n_jobs left alone): at n~56 rows thread overhead costs more than
    the trees, measured 5.89s vs 4.54s.
    """
    rows = []
    for k in range(1, len(FEATURES) + 1):
        batch = list(combinations(FEATURES, k))
        if jobs > 1:
            rows += Parallel(n_jobs=jobs)(
                delayed(score_subset)(X, y, folds, cols) for cols in batch)
        else:
            rows += [score_subset(X, y, folds, cols) for cols in batch]
        print(f"   ...{k} feature(s): {len(rows)} subsets scored", flush=True)
    return pd.DataFrame(rows)


def greedy_path(df: pd.DataFrame, metric: str) -> list[dict]:
    """The story Aruna asked for: add the feature that gains the most, one at a time."""
    chosen: list[str] = []
    path = []
    for _ in FEATURES:
        best = None
        for f in FEATURES:
            if f in chosen:
                continue
            key = "+".join(sorted(chosen + [f], key=FEATURES.index))
            row = df[df.features == key]
            if len(row) and (best is None or row.iloc[0][metric] > best[1]):
                best = (f, float(row.iloc[0][metric]), row.iloc[0])
        if best is None:
            break
        chosen.append(best[0])
        path.append({"added": best[0], "n": len(chosen), metric: best[1],
                     "gain": best[1] - (path[-1][metric] if path else 0.0),
                     "set": "+".join(chosen)})
    return path


# -------------------------------------------------------------- nested check

def _inner_score(Xtr, ytr, ftr, cols, metric: str) -> float:
    if metric == "vote":
        return cv_vote_rule(Xtr, ytr, ftr, list(cols))[0]
    return cv_macro_f1(Xtr[list(cols)], ytr, ftr, rf)


def nested_estimate(X, y, folds, metric: str, jobs: int = 1) -> tuple[float, list[str]]:
    """Re-select the best subset inside every outer fold, score on the unseen fold."""
    pred = np.empty(len(y), dtype=object)
    picked = []
    all_cols = [c for size in range(1, len(FEATURES) + 1)
                for c in combinations(FEATURES, size)]
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        Xtr, ytr, ftr = X[tr], y[tr], folds[tr]
        if jobs > 1:
            scores = Parallel(n_jobs=jobs)(
                delayed(_inner_score)(Xtr, ytr, ftr, cols, metric) for cols in all_cols)
        else:
            scores = [_inner_score(Xtr, ytr, ftr, cols, metric) for cols in all_cols]
        # First maximum wins, matching the old strictly-greater scan over the same order.
        best_cols = all_cols[int(np.argmax(scores))]
        picked.append("+".join(best_cols))
        if metric == "vote":
            rules, v = fit_vote_rule(Xtr[list(best_cols)], ytr)
            pred[te] = apply_vote_rule(X.loc[te, list(best_cols)], rules, v)
        else:
            m = rf().fit(Xtr[list(best_cols)], ytr)
            pred[te] = m.predict(X.loc[te, list(best_cols)])
        print(f"   outer fold {int(k)}: picked {picked[-1]}", flush=True)
    return f1_score(y, pred, average="macro"), picked


# --------------------------------------------------------------------- data

def load(args) -> tuple[pd.DataFrame, str]:
    if args.tracker:
        path = RES / f"tracker_features_{args.tracker}.csv"
        df = pd.read_csv(path)
        df = df[df.policy == args.policy]
        return df, f"{args.tracker}_{args.policy}"
    return pd.read_csv(RES / "ceiling_features_track.csv"), "ceiling"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", default=None, help="omit for the perfect-track ceiling")
    ap.add_argument("--policy", default="longest", choices=["oracle", "longest"])
    ap.add_argument("--classes", nargs="+", default=["BIRD", "DRONE"])
    ap.add_argument("--nested", action="store_true", help="run the selection honesty check")
    ap.add_argument("--nested-only", action="store_true", dest="nested_only",
                    help="skip re-scoring the 127 subsets and read the existing "
                         "feature_selection_<unit>.csv instead; implies --nested. The flat "
                         "table is deterministic, so re-running it only costs time.")
    ap.add_argument("--jobs", type=int, default=6,
                    help="processes for the subset sweep (6P+4E cores here; the table "
                         "does not depend on this)")
    args = ap.parse_args()

    df, unit = load(args)
    df = df[df.cls.isin(args.classes)].reset_index(drop=True)
    y, X, folds = df.cls.values, df[FEATURES], df.fold.values
    print(f"unit: {unit} | {len(df)} tracks ({dict(df.cls.value_counts())}) | "
          f"{len(np.unique(folds))} folds\n")

    f1_base, acc_base = cv_majority(y, folds)
    print("0) BASELINE, NO FEATURES (majority class)")
    print(f"   accuracy {acc_base:.1%} | macro F1 {f1_base:.3f}\n")

    cached = RES / f"feature_selection_{unit}.csv"
    if args.nested_only:
        if not cached.exists():
            print(f"--nested-only needs {cached}, which does not exist. Run without it first.")
            return 1
        args.nested = True
        res = pd.read_csv(cached)
        print(f"1) REUSING {len(res)} ALREADY-SCORED SUBSETS from {cached.name}")
    else:
        print(f"1) SCORING ALL {2 ** len(FEATURES) - 1} SUBSETS")
        res = all_subsets(X, y, folds, args.jobs)
        res.to_csv(cached, index=False)

    print("\n2) EACH FEATURE ALONE (sorted by random forest)")
    solo = res[res.k == 1].sort_values("rf_macro_f1", ascending=False)
    print(f"   {'feature':12s}{'vote F1':>10}{'vote acc':>10}{'RF F1':>9}{'RF acc':>9}"
          f"{'RF gain':>9}")
    for _, r in solo.iterrows():
        print(f"   {r.features:12s}{r.vote_macro_f1:>10.3f}{r.vote_accuracy:>10.1%}"
              f"{r.rf_macro_f1:>9.3f}{r.rf_accuracy:>9.1%}{r.rf_macro_f1 - f1_base:>+9.3f}")

    print("\n3) BEST SUBSET OF EACH SIZE")
    print(f"   {'k':>2} {'best by RF':48s}{'RF F1':>8}{'gain':>8}   "
          f"{'best by vote':38s}{'vote F1':>9}")
    prev_rf = prev_v = f1_base
    for k in range(1, len(FEATURES) + 1):
        sub = res[res.k == k]
        b_rf = sub.loc[sub.rf_macro_f1.idxmax()]
        b_v = sub.loc[sub.vote_macro_f1.idxmax()]
        print(f"   {k:>2} {b_rf.features:48s}{b_rf.rf_macro_f1:>8.3f}"
              f"{b_rf.rf_macro_f1 - prev_rf:>+8.3f}   {b_v.features:38s}"
              f"{b_v.vote_macro_f1:>9.3f}")
        prev_rf, prev_v = b_rf.rf_macro_f1, b_v.vote_macro_f1

    print("\n4) GREEDY PATH -- add the feature that gains most (random forest)")
    run = f1_base
    for step in greedy_path(res, "rf_macro_f1"):
        print(f"   +{step['added']:12s} n={step['n']}  F1 {step['rf_macro_f1']:.3f}  "
              f"({step['rf_macro_f1'] - run:+.3f})")
        run = step["rf_macro_f1"]

    print("\n5) THE WINNERS")
    w_rf = res.loc[res.rf_macro_f1.idxmax()]
    w_v = res.loc[res.vote_macro_f1.idxmax()]
    full = res[res.features == "+".join(FEATURES)].iloc[0]
    print(f"   random forest : {w_rf.features}  F1 {w_rf.rf_macro_f1:.3f} "
          f"(all 7 = {full.rf_macro_f1:.3f}, {w_rf.rf_macro_f1 - full.rf_macro_f1:+.3f})")
    print(f"   voting rule   : {w_v.features}  F1 {w_v.vote_macro_f1:.3f} "
          f"(all 7 = {full.vote_macro_f1:.3f}, {w_v.vote_macro_f1 - full.vote_macro_f1:+.3f})")

    if args.nested:
        print("\n6) HONESTY CHECK -- selection repeated inside each outer fold")
        for metric, label in (("rf", "random forest"), ("vote", "voting rule")):
            score, picked = nested_estimate(X, y, folds, metric, args.jobs)
            flat = res.loc[res[f"{metric}_macro_f1"].idxmax()][f"{metric}_macro_f1"]
            print(f"   {label:14s} nested {score:.3f} vs best-of-all {flat:.3f} "
                  f"({score - flat:+.3f})")
            print(f"   {'':14s} folds picked: {picked}")

    print(f"\n-> {RES / f'feature_selection_{unit}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
