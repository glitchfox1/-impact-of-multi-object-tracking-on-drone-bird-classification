#!/usr/bin/env python3
"""Statistics for the Paper 1 manuscript -- analysis only.

Reads the feature and quality CSVs that feature_ceiling.py and tracker_features.py wrote, and
adds what a reviewer needs before any comparison can be called a difference:

  1. out-of-fold predictions for every unit (ceiling, 9 trackers x 2 policies), on the full
     clip set and on the clips all nine trackers kept (the "common" set, ADR-003)
  2. 95% confidence intervals for macro F1 -- clip-level bootstrap, stratified by class
  3. paired tests on the common set: every tracker against the best tracker and against the
     ceiling -- paired bootstrap of the F1 difference and an exact McNemar test, both
     Holm-corrected within each family
  4. class-dependent fragmentation -- fragments per ground-truth object, bird vs drone,
     Mann-Whitney U per tracker, Holm-corrected
  5. the hovering confound -- does tortuosity carry anything once net displacement per
     second is known?

Folds, classifiers and the voting rule are IMPORTED from feature_ceiling / feature_selection,
so every macro F1 here must reproduce tracker_classification.csv / ceiling_summary_track.csv
to 4 dp; main() checks that before writing anything.

    python src/paper1/paper_stats.py            # B = 10,000 bootstrap resamples, seed 0

Writes results/paper_oof_predictions.csv and results/paper_stats_{ci,paired,fragmentation,
tortuosity}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_ceiling import FEATURES, apply_vote_rule, fit_vote_rule  # noqa: E402
from feature_selection import rf_cv_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
TRACKERS = ["bytetrack", "occluboost", "botsort", "strongsort", "ocsort", "hybridsort",
            "boosttrack", "deepocsort", "sfsort"]
POLICIES = ["longest", "oracle"]
CLASSES = ["BIRD", "DRONE"]


# ------------------------------------------------------------------- loading

def load_units() -> dict[tuple[str, str], pd.DataFrame]:
    """(tracker, policy) -> one row per clip. The ceiling is ("ceiling", "gt")."""
    ceil = pd.read_csv(RES / "ceiling_features_track.csv")
    units = {("ceiling", "gt"): ceil[ceil.cls.isin(CLASSES)].reset_index(drop=True)}
    for t in TRACKERS:
        df = pd.read_csv(RES / f"tracker_features_{t}.csv")
        for p in POLICIES:
            units[(t, p)] = df[(df.policy == p) & df.cls.isin(CLASSES)].reset_index(drop=True)
    return units


def common_videos(units: dict, policy: str) -> set[str]:
    """Clips every one of the nine trackers kept under this policy (ADR-003) AND that have a
    usable ground-truth track (ADR-006), so the matched ceiling is scored on identical clips."""
    sets = [set(units[(t, policy)].video) for t in TRACKERS] + [set(units[("ceiling", "gt")].video)]
    return set.intersection(*sets)


# ------------------------------------------------------------- predictions

def vote_cv_predict(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions of the fitted voting rule (cv_vote_rule without the scoring)."""
    pred = np.empty(len(y), dtype=object)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        rules, v = fit_vote_rule(X.loc[tr], y[tr])
        pred[te] = apply_vote_rule(X.loc[te], rules, v)
    return pred


def oof(df: pd.DataFrame) -> pd.DataFrame:
    y, X, folds = df.cls.values, df[FEATURES], df.fold.values
    out = df[["video", "cls", "fold"]].copy()
    out["rf_pred"] = rf_cv_predict(X, y, folds, FEATURES)
    out["vote_pred"] = vote_cv_predict(X, y, folds)
    return out


# ---------------------------------------------------------------- bootstrap

def f1_from_idx(yb: np.ndarray, pb: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro F1 (bird, drone) for every row of an index matrix -- vectorised over resamples."""
    y, p = yb[idx], pb[idx]
    tp_b = (y & p).sum(1)
    fp_b = (~y & p).sum(1)
    fn_b = (y & ~p).sum(1)
    tp_d = (~y & ~p).sum(1)
    f1_b = np.where(2 * tp_b + fp_b + fn_b > 0, 2 * tp_b / np.maximum(2 * tp_b + fp_b + fn_b, 1), 0)
    f1_d = np.where(2 * tp_d + fn_b + fp_b > 0, 2 * tp_d / np.maximum(2 * tp_d + fn_b + fp_b, 1), 0)
    return (f1_b + f1_d) / 2


def strat_index(yb: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """B stratified resamples (birds and drones resampled separately, class sizes kept)."""
    bird, drone = np.where(yb)[0], np.where(~yb)[0]
    return np.hstack([rng.choice(bird, (B, len(bird))), rng.choice(drone, (B, len(drone)))])


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, same order as the input."""
    order = np.argsort(p)
    m, adj, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()


# ------------------------------------------------------------------- tables

def ci_table(preds: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    rows = []
    for (t, p, s), g in preds.groupby(["tracker", "policy", "set"], sort=False):
        yb = (g.cls.values == "BIRD")
        idx = strat_index(yb, B, np.random.default_rng(seed))
        for rule in ("rf", "vote"):
            pb = (g[f"{rule}_pred"].values == "BIRD")
            point = f1_score(g.cls, g[f"{rule}_pred"], average="macro")
            boot = f1_from_idx(yb, pb, idx)
            rows.append({"tracker": t, "policy": p, "set": s, "rule": rule, "n": len(g),
                         "n_bird": int(yb.sum()), "n_drone": int((~yb).sum()),
                         "macro_f1": round(point, 4),
                         "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                         "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
                         "bird_recall": round(float((pb & yb).sum() / yb.sum()), 4),
                         "drone_recall": round(float((~pb & ~yb).sum() / (~yb).sum()), 4),
                         "accuracy": round(float((pb == yb).mean()), 4)})
    return pd.DataFrame(rows)


def paired_table(preds: pd.DataFrame, ci: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """Every tracker vs (a) the best tracker and (b) the ceiling, on the common set."""
    rows = []
    for policy in POLICIES:
        # the ceiling's common set carries the policy it is matched to (see main)
        com = preds[(preds.set == "common") & (preds.policy == policy)]
        for rule in ("rf", "vote"):
            sub = ci[(ci.set == "common") & (ci.policy == policy) & (ci.rule == rule) &
                     (ci.tracker != "ceiling")]
            best = sub.loc[sub.macro_f1.idxmax(), "tracker"]
            for ref_kind, ref in (("best", best), ("ceiling", "ceiling")):
                a = com[com.tracker == ref].set_index("video")
                fam = []
                for t in TRACKERS:
                    if t == ref:
                        continue
                    b = com[com.tracker == t].set_index("video")
                    b = b.loc[a.index]                       # pair by clip
                    yb = (a.cls.values == "BIRD")
                    pa, pb_ = a[f"{rule}_pred"].values == "BIRD", b[f"{rule}_pred"].values == "BIRD"
                    idx = strat_index(yb, B, np.random.default_rng(seed))
                    d = f1_from_idx(yb, pb_, idx) - f1_from_idx(yb, pa, idx)
                    ok_a, ok_b = pa == yb, pb_ == yb
                    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
                    p_mc = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                    f1a = f1_score(a.cls, a[f"{rule}_pred"], average="macro")
                    f1b = f1_score(b.cls, b[f"{rule}_pred"], average="macro")
                    p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                    fam.append({"policy": policy, "rule": rule, "reference": ref_kind,
                                "a": ref, "b": t, "n": len(a),
                                "f1_a": round(f1a, 4), "f1_b": round(f1b, 4),
                                "delta_b_minus_a": round(f1b - f1a, 4),
                                "delta_ci_lo": round(float(np.percentile(d, 2.5)), 4),
                                "delta_ci_hi": round(float(np.percentile(d, 97.5)), 4),
                                "p_boot": round(float(p_boot), 4),
                                "a_right_b_wrong": n01, "a_wrong_b_right": n10,
                                "p_mcnemar": round(float(p_mc), 4)})
                for key in ("p_boot", "p_mcnemar"):
                    for r, adj in zip(fam, holm([r[key] for r in fam])):
                        r[f"{key}_holm"] = round(adj, 4)
                rows += fam
    return pd.DataFrame(rows)


QUALITY_METRICS = {                    # tracker_quality.csv column -> short name
    "n_tracks_on_gt": "frag",           # tracks overlapping the GT object (fragments)
    "oracle_coverage": "coverage",      # share of GT frames the best-matched track covers
    "det_recall": "recall",             # share of GT boxes matched by ANY tracker output box
    "id_switches": "idsw",
}


def fragmentation_table() -> pd.DataFrame:
    """Bird vs drone, per tracker, for every tracking-quality metric.

    Mann-Whitney U per tracker and metric; Holm across the nine trackers WITHIN a metric.
    rank_biserial > 0 means birds score higher than drones.
    """
    q = pd.read_csv(RES / "tracker_quality.csv")
    q = q[q.cls.isin(CLASSES)]
    rows = {t: {"tracker": t} for t in TRACKERS}
    for col, short in QUALITY_METRICS.items():
        pvals = []
        for t in TRACKERS:
            g = q[q.tracker == t]
            bird, drone = g[g.cls == "BIRD"][col], g[g.cls == "DRONE"][col]
            u, p = mannwhitneyu(bird, drone, alternative="two-sided")
            rows[t].update({"n_bird": len(bird), "n_drone": len(drone),
                            f"{short}_bird_mean": round(bird.mean(), 3),
                            f"{short}_drone_mean": round(drone.mean(), 3),
                            f"{short}_bird_median": round(float(bird.median()), 3),
                            f"{short}_drone_median": round(float(drone.median()), 3),
                            f"{short}_rank_biserial": round(2 * u / (len(bird) * len(drone)) - 1, 3),
                            f"{short}_p": round(float(p), 6)})
            pvals.append(float(p))
        for t, adj in zip(TRACKERS, holm(pvals)):
            rows[t][f"{short}_p_holm"] = round(adj, 6)
    return pd.DataFrame(list(rows.values()))


def tortuosity_table(units: dict) -> pd.DataFrame:
    """Is tortuosity a behaviour feature, or a hovering detector? (plan Step 2)"""
    rows = []
    six = [f for f in FEATURES if f != "tortuosity"]
    for (t, p), df in units.items():
        if p == "oracle":
            continue
        y, folds = df.cls.values, df.fold.values
        yb = (y == "BIRD").astype(int)

        def f1(cols):
            return round(f1_score(y, rf_cv_predict(df, y, folds, cols), average="macro"), 4)

        rows.append({
            "tracker": t, "policy": p, "n": len(df),
            "tort_lt1_bird": int(((df.tortuosity < 1) & (y == "BIRD")).sum()),
            "tort_lt1_drone": int(((df.tortuosity < 1) & (y == "DRONE")).sum()),
            "tort_cap_bird": int(((df.tortuosity >= 10) & (y == "BIRD")).sum()),
            "tort_cap_drone": int(((df.tortuosity >= 10) & (y == "DRONE")).sum()),
            "tort_median_bird": round(df.tortuosity[y == "BIRD"].median(), 3),
            "tort_median_drone": round(df.tortuosity[y == "DRONE"].median(), 3),
            "disp_median_bird": round(df.net_disp_s[y == "BIRD"].median(), 4),
            "disp_median_drone": round(df.net_disp_s[y == "DRONE"].median(), 4),
            "auc_tortuosity": round(roc_auc_score(yb, df.tortuosity), 4),
            "auc_net_disp_s": round(roc_auc_score(yb, df.net_disp_s), 4),
            "spearman_tort_disp": round(df.tortuosity.corr(df.net_disp_s, method="spearman"), 3),
            "rf_tortuosity_only": f1(["tortuosity"]),
            "rf_net_disp_only": f1(["net_disp_s"]),
            "rf_tort_plus_disp": f1(["tortuosity", "net_disp_s"]),
            "rf_all7": f1(FEATURES),
            "rf_six_no_tort": f1(six),
            "rf_six_plus_disp": f1(six + ["net_disp_s"]),
        })
    return pd.DataFrame(rows)


def leakage_table(units: dict) -> pd.DataFrame:
    """Does tortuosity on real tracks encode track TRUNCATION rather than flight?

    completeness = selected track's frames / the clip's ground-truth frames. If bird tracks are
    cut short (class-dependent tracking failure), a short piece of a bird's flight is nearly
    straight, so bird tortuosity would rise with completeness.
    """
    q = pd.read_csv(RES / "tracker_quality.csv")[["tracker", "video", "gt_frames"]]
    ceil = units[("ceiling", "gt")].set_index("video")
    rows = []
    for (t, p), df in units.items():
        if p != "longest":
            continue
        # completeness needs a GT track, so it is scored on the clips that have one (ADR-006);
        # track length needs no GT and is scored on the tracker's full clip set, like every
        # other longest-policy number
        d = df.merge(q[q.tracker == t], on="video")
        d["completeness"] = (d.n_frames / d.gt_frames).clip(upper=1.0)
        y, folds = d.cls.values, d.fold.values
        y_all, folds_all = df.cls.values, df.fold.values
        bird, drone = d[d.cls == "BIRD"], d[d.cls == "DRONE"]
        g = bird.set_index("video").tortuosity
        both = g.index.intersection(ceil.index)
        u, pc = mannwhitneyu(bird.completeness, drone.completeness, alternative="two-sided")
        rows.append({
            "tracker": t, "n": len(d),
            "completeness_bird_median": round(bird.completeness.median(), 3),
            "completeness_drone_median": round(drone.completeness.median(), 3),
            "p_completeness": round(float(pc), 6),
            "spearman_tort_completeness_bird": round(bird.tortuosity.corr(bird.completeness, method="spearman"), 3),
            "spearman_tort_completeness_drone": round(drone.tortuosity.corr(drone.completeness, method="spearman"), 3),
            "bird_tort_ratio_track_vs_gt_median": round(float((g[both] / ceil.loc[both, "tortuosity"]).median()), 3),
            "n_all": len(df),
            "rf_n_frames_only": round(f1_score(y_all, rf_cv_predict(df, y_all, folds_all, ["n_frames"]),
                                               average="macro"), 4),
            "rf_completeness_only": round(f1_score(y, rf_cv_predict(d, y, folds, ["completeness"]), average="macro"), 4),
        })
    out = pd.DataFrame(rows)
    out["p_completeness_holm"] = [round(a, 6) for a in holm(out.p_completeness.tolist())]
    return out


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality-only", action="store_true", dest="quality_only",
                    help="recompute only paper_stats_fragmentation.csv (no model fitting)")
    ap.add_argument("--leakage-only", action="store_true", dest="leakage_only",
                    help="recompute only paper_stats_leakage.csv (track truncation vs tortuosity)")
    args = ap.parse_args()

    ap_only = args.quality_only or args.leakage_only
    if args.quality_only:
        frag = fragmentation_table()
        frag.to_csv(RES / "paper_stats_fragmentation.csv", index=False)
        print(frag.to_string(index=False))
    if args.leakage_only:
        leak = leakage_table(load_units())
        leak.to_csv(RES / "paper_stats_leakage.csv", index=False)
        print(leak.to_string(index=False))
    if ap_only:
        return 0

    units = load_units()
    common = {p: common_videos(units, p) for p in POLICIES}
    print("common clips: " + ", ".join(f"{p} {len(v)}" for p, v in common.items()))

    # Rows are labelled (tracker, policy, set). The ceiling's full set is policy "gt"; it has
    # one common set per policy (the clips all nine trackers kept under THAT policy), each
    # labelled with that policy, so "ceiling / longest / common" is the matched ceiling.
    frames = []
    for (t, p), df in units.items():
        jobs = [(p, "full", df)]
        for cp in (POLICIES if t == "ceiling" else [p]):
            jobs.append((cp, "common", df[df.video.isin(common[cp])].reset_index(drop=True)))
        for pol, s, d in jobs:
            o = oof(d)
            o.insert(0, "set", s)
            o.insert(0, "policy", pol)
            o.insert(0, "tracker", t)
            frames.append(o)
    preds = pd.concat(frames, ignore_index=True)

    # --- consistency: must reproduce the pipeline's own tables to 4 dp
    tc = pd.read_csv(RES / "tracker_classification.csv").set_index("unit")
    cs = pd.read_csv(RES / "ceiling_summary_track.csv").set_index("metric").value
    bad = []
    for (t, p), g in preds[preds.set == "full"].groupby(["tracker", "policy"]):
        rf = round(f1_score(g.cls, g.rf_pred, average="macro"), 4)
        vote = round(f1_score(g.cls, g.vote_pred, average="macro"), 4)
        if t == "ceiling":
            ref_rf, ref_vote = float(cs["cv_macro_f1_random forest"]), float(cs["vote_rule_fitted_macro_f1"])
        else:
            ref_rf = tc.loc[f"{t}/{p}", "rf_macro_f1"]
            ref_vote = tc.loc[f"{t}/{p}", "fitted_vote_macro_f1"]
        if abs(rf - ref_rf) > 1e-4 or abs(vote - ref_vote) > 1e-4:
            bad.append(f"{t}/{p}: rf {rf} vs {ref_rf}, vote {vote} vs {ref_vote}")
    if bad:
        print("CONSISTENCY FAILED -- nothing written:\n  " + "\n  ".join(bad))
        return 1
    print("consistency: all full-set RF and vote macro F1 reproduce the pipeline tables (4 dp)")

    preds.to_csv(RES / "paper_oof_predictions.csv", index=False)

    ci = ci_table(preds, args.boot, args.seed)
    ci.to_csv(RES / "paper_stats_ci.csv", index=False)
    assert ((ci.ci_lo <= ci.macro_f1) & (ci.macro_f1 <= ci.ci_hi)).all(), "point outside its CI"

    paired = paired_table(preds, ci, args.boot, args.seed)
    paired.to_csv(RES / "paper_stats_paired.csv", index=False)
    assert ((paired.a_right_b_wrong + paired.a_wrong_b_right) <= paired.n).all()

    frag = fragmentation_table()
    frag.to_csv(RES / "paper_stats_fragmentation.csv", index=False)
    tort = tortuosity_table(units)
    tort.to_csv(RES / "paper_stats_tortuosity.csv", index=False)
    leak = leakage_table(units)
    leak.to_csv(RES / "paper_stats_leakage.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nCI (random forest, longest + ceiling):")
    print(ci[(ci.rule == "rf") & (ci.policy.isin(["longest", "gt"]))].to_string(index=False))
    print("\nPAIRED (random forest, longest):")
    print(paired[(paired.rule == "rf") & (paired.policy == "longest")].to_string(index=False))
    print("\nFRAGMENTATION:")
    print(frag.to_string(index=False))
    print("\nTORTUOSITY / HOVERING:")
    print(tort.to_string(index=False))
    print("\n-> results/paper_oof_predictions.csv, paper_stats_{ci,paired,fragmentation,tortuosity}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
