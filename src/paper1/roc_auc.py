#!/usr/bin/env python3
"""ROC curves and AUC for the Paper 1 manuscript -- analysis only.

paper_stats.py keeps only hard out-of-fold labels, so a ROC curve needs the scores behind
them. This script refits the SAME models on the SAME folds and keeps the score:

  * fitted voting rule -- the number of features voting "bird" (0-7), with each fold's
    thresholds learned on its training folds (feature_ceiling.fit_vote_rule)
  * random forest      -- out-of-fold P(bird) (300 trees, random_state 0, as everywhere)

Scores are pooled over the five folds. Before anything is written, the scores are turned
back into labels exactly as the pipeline does (votes >= the fold's fitted count; the
forest's own predict) and must reproduce results/paper_oof_predictions.csv clip for clip.

AUC carries a 95% clip-level bootstrap interval (stratified by class, as paper_stats.py),
and on the common clips every tracker is compared with the perfect-track reference by a
paired bootstrap of the AUC difference, Holm-corrected across the nine trackers.

    python src/paper1/roc_auc.py            # B = 10,000 bootstrap resamples, seed 0
    python src/paper1/roc_auc.py --path-ablation

Writes results/roc_auc.csv, results/roc_auc_paired.csv and results/roc_curves.csv.

--path-ablation instead asks whether the path-group result of the manuscript (Table 5, macro
F1) also holds for AUC: on the common clips (longest policy), every unit is scored with all
seven features and without the path group, and the AUC change carries a paired bootstrap
interval. Its gate: the forest's macro F1 without the path group must reproduce rf_minus_path
in results/tracker_classification_common.csv. Writes results/roc_auc_path_ablation.csv.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_ceiling import FEATURES, GROUPS, fit_vote_rule  # noqa: E402
from feature_selection import rf  # noqa: E402
from paper_stats import (  # noqa: E402
    POLICIES, RES, TRACKERS, common_videos, holm, load_units, strat_index,
)

RULES = ("vote", "rf")                 # voting rule first, as in the manuscript


def vote_counts(X: pd.DataFrame, rules: dict) -> np.ndarray:
    return np.array([sum((X.iloc[i][f] > t) if s == 1 else (X.iloc[i][f] < t)
                         for f, (t, s) in rules.items()) for i in range(len(X))])


def oof_scores(df: pd.DataFrame, cols: list[str] = FEATURES) -> pd.DataFrame:
    """Out-of-fold scores AND the labels the pipeline derives from them."""
    y, X, folds = df.cls.values, df[cols], df.fold.values
    out = df[["video", "cls", "fold"]].copy()
    vote_s, vote_p = np.empty(len(y)), np.empty(len(y), dtype=object)
    rf_s, rf_p = np.empty(len(y)), np.empty(len(y), dtype=object)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        rules, v = fit_vote_rule(X.loc[tr], y[tr])
        c = vote_counts(X.loc[te], rules)
        vote_s[te], vote_p[te] = c, np.where(c >= v, "BIRD", "DRONE")
        m = rf().fit(X.loc[tr], y[tr])
        rf_s[te] = m.predict_proba(X.loc[te])[:, list(m.classes_).index("BIRD")]
        rf_p[te] = m.predict(X.loc[te])
    out["vote_score"], out["vote_pred"] = vote_s, vote_p
    out["rf_score"], out["rf_pred"] = rf_s, rf_p
    return out


def auc_from_idx(yb: np.ndarray, s: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """AUC (Mann-Whitney form, ties count one half) for every row of an index matrix.
    strat_index puts the birds first in each row, so the class sizes are fixed."""
    n_b, n_d = int(yb.sum()), int((~yb).sum())
    r = rankdata(s[idx], axis=1)
    return (r[:, :n_b].sum(1) - n_b * (n_b + 1) / 2) / (n_b * n_d)


def path_ablation(units: dict, common: set, B: int, seed: int) -> int:
    """All seven features vs without the path group, per unit, on the common clips (longest)."""
    keep = [f for f in FEATURES if f not in GROUPS["path"]]
    tc = pd.read_csv(RES / "tracker_classification_common.csv")
    tc = tc[(tc.policy == "longest") & (tc.set == "common")].set_index("tracker")
    rows, bad = [], []
    for t in ["ceiling"] + TRACKERS:
        df = units[(t, "gt" if t == "ceiling" else "longest")]
        d = df[df.video.isin(common)].reset_index(drop=True)
        full, nop = oof_scores(d), oof_scores(d, keep)
        yb = d.cls.values == "BIRD"
        f1_drop = f1_score(d.cls, nop.rf_pred, average="macro") - f1_score(d.cls, full.rf_pred, average="macro")
        if abs(round(f1_drop, 4) - tc.loc[t, "rf_minus_path"]) > 1e-4:
            bad.append(f"{t}: rf_minus_path {f1_drop:.4f} vs {tc.loc[t, 'rf_minus_path']}")
        idx = strat_index(yb, B, np.random.default_rng(seed))
        for rule in RULES:
            sa, sb = full[f"{rule}_score"].values.astype(float), nop[f"{rule}_score"].values.astype(float)
            diff = auc_from_idx(yb, sb, idx) - auc_from_idx(yb, sa, idx)
            a_all, a_nop = roc_auc_score(yb, sa), roc_auc_score(yb, sb)
            rows.append({"tracker": t, "rule": rule, "n": len(d),
                         "auc_all7": round(a_all, 4), "auc_without_path": round(a_nop, 4),
                         "change": round(a_nop - a_all, 4),
                         "change_ci_lo": round(float(np.percentile(diff, 2.5)), 4),
                         "change_ci_hi": round(float(np.percentile(diff, 97.5)), 4)})
    if bad:
        print("GATE FAILED -- nothing written:\n  " + "\n  ".join(bad))
        return 1
    print("gate: RF macro-F1 change without the path group reproduces rf_minus_path for all 10 units")
    out = pd.DataFrame(rows)
    out.to_csv(RES / "roc_auc_path_ablation.csv", index=False)
    print(out.to_string(index=False))
    print("-> results/roc_auc_path_ablation.csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--path-ablation", action="store_true", dest="path_ablation")
    args = ap.parse_args()
    t0 = datetime.now()
    print(f"start {t0:%Y-%m-%d %H:%M:%S}")

    units = load_units()
    common = {p: common_videos(units, p) for p in POLICIES}
    if args.path_ablation:
        rc = path_ablation(units, common["longest"], args.boot, args.seed)
        t1 = datetime.now()
        print(f"end {t1:%Y-%m-%d %H:%M:%S} (took {t1 - t0})")
        return rc
    frames = []
    for (t, p), df in units.items():            # same unit/set labelling as paper_stats.main
        jobs = [(p, "full", df)]
        for cp in (POLICIES if t == "ceiling" else [p]):
            jobs.append((cp, "common", df[df.video.isin(common[cp])].reset_index(drop=True)))
        for pol, s, d in jobs:
            o = oof_scores(d)
            o.insert(0, "set", s)
            o.insert(0, "policy", pol)
            o.insert(0, "tracker", t)
            frames.append(o)
    scores = pd.concat(frames, ignore_index=True)

    # --- sanity gate: the labels must be paper_stats.py's labels, clip for clip
    key = ["tracker", "policy", "set", "video"]
    ref = pd.read_csv(RES / "paper_oof_predictions.csv")
    m = scores.merge(ref, on=key, suffixes=("", "_paper"), how="outer", indicator=True)
    bad = m[(m._merge != "both") | (m.vote_pred != m.vote_pred_paper) | (m.rf_pred != m.rf_pred_paper)]
    if len(bad):
        print(f"SANITY GATE FAILED -- {len(bad)} rows differ from paper_oof_predictions.csv; "
              "nothing written:")
        print(bad[key + ["vote_pred", "vote_pred_paper", "rf_pred", "rf_pred_paper", "_merge"]]
              .head(20).to_string(index=False))
        return 1
    print(f"sanity gate: all {len(scores)} rows reproduce paper_oof_predictions.csv "
          "(vote and rf labels)")

    rows, curves = [], []
    for (t, p, s), g in scores.groupby(["tracker", "policy", "set"], sort=False):
        yb = g.cls.values == "BIRD"
        idx = strat_index(yb, args.boot, np.random.default_rng(args.seed))
        for rule in RULES:
            sc = g[f"{rule}_score"].values.astype(float)
            point = roc_auc_score(yb, sc)
            boot = auc_from_idx(yb, sc, idx)
            rows.append({"tracker": t, "policy": p, "set": s, "rule": rule, "n": len(g),
                         "n_bird": int(yb.sum()), "n_drone": int((~yb).sum()),
                         "auc": round(point, 4),
                         "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                         "ci_hi": round(float(np.percentile(boot, 97.5)), 4)})
            fpr, tpr, thr = roc_curve(yb, sc)
            curves.append(pd.DataFrame({"tracker": t, "policy": p, "set": s, "rule": rule,
                                        "fpr": fpr.round(5), "tpr": tpr.round(5),
                                        "threshold": thr}))
    auc = pd.DataFrame(rows)
    assert ((auc.ci_lo <= auc.auc) & (auc.auc <= auc.ci_hi)).all(), "point outside its CI"

    # --- paired: every tracker vs the perfect-track reference on the same common clips
    paired = []
    for policy in POLICIES:
        com = scores[(scores.set == "common") & (scores.policy == policy)]
        a = com[com.tracker == "ceiling"].set_index("video")
        yb = a.cls.values == "BIRD"
        idx = strat_index(yb, args.boot, np.random.default_rng(args.seed))
        for rule in RULES:
            sa = a[f"{rule}_score"].values.astype(float)
            ba = auc_from_idx(yb, sa, idx)
            fam = []
            for t in TRACKERS:
                b = com[com.tracker == t].set_index("video").loc[a.index]   # pair by clip
                sb = b[f"{rule}_score"].values.astype(float)
                d = auc_from_idx(yb, sb, idx) - ba
                p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                fam.append({"policy": policy, "rule": rule, "a": "ceiling", "b": t,
                            "n": len(a), "auc_a": round(roc_auc_score(yb, sa), 4),
                            "auc_b": round(roc_auc_score(yb, sb), 4),
                            "delta_b_minus_a": round(roc_auc_score(yb, sb) - roc_auc_score(yb, sa), 4),
                            "delta_ci_lo": round(float(np.percentile(d, 2.5)), 4),
                            "delta_ci_hi": round(float(np.percentile(d, 97.5)), 4),
                            "p_boot": round(float(p_boot), 4)})
            for r, adj in zip(fam, holm([r["p_boot"] for r in fam])):
                r["p_boot_holm"] = round(adj, 4)
            paired += fam
    paired = pd.DataFrame(paired)

    auc.to_csv(RES / "roc_auc.csv", index=False)
    paired.to_csv(RES / "roc_auc_paired.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RES / "roc_curves.csv", index=False)

    pd.set_option("display.width", 200)
    show = auc[(auc.policy.isin(["longest", "gt"]))]
    print("\nAUC (longest policy + reference):")
    print(show.to_string(index=False))
    print("\nPAIRED vs perfect-track reference (longest, common clips):")
    print(paired[paired.policy == "longest"].to_string(index=False))
    t1 = datetime.now()
    print(f"\n-> results/roc_auc.csv, roc_auc_paired.csv, roc_curves.csv\n"
          f"end {t1:%Y-%m-%d %H:%M:%S} (took {t1 - t0})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
