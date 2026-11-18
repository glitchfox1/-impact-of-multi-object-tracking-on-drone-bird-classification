#!/usr/bin/env python3
"""Aggregate the per-unit feature_selection_*.csv tables into the cross-tracker answer.

feature_selection.py answers Aruna's question for ONE unit (one tracker+policy, or the
perfect-track ceiling). This script reads every unit it finds and answers the question she
actually asked last: what is the best combination of the 7 voting features TOGETHER WITH
the tracking, and how does the picture differ between perfect tracks and real ones?

Three tables, all written to results/ and echoed to stdout:
  1. feature_selection_summary.csv -- one row per unit: baseline, best single feature,
     best subset, all-7, and the margin the selection buys over using everything.
  2. feature_selection_solo.csv    -- unit x feature, each feature ALONE. This is the
     "add one feature to the bare pipeline and see the change" table.
  3. feature_selection_entry.csv   -- unit x feature, the greedy step at which each
     feature first enters. Low number = the feature earns its place early.
  4. feature_selection_nested.csv  -- unit x rule, the HONEST figures: best-of-127 (biased),
     nested (selection redone inside each outer fold), all-7, and how often each feature was
     picked across the 5 outer folds. feature_selection.py --nested prints these but never
     writes them to a CSV, so they are parsed from its logs (logs/fs_*_nested.log for the
     two units nested on 2026-09-20, logs/fsn_<unit>.log for the rest). Quote THIS table's
     nested column, never the best-of-127 one.

The flip this is built to expose: at the ceiling the motion features carry the result,
on real tracker output they do not (detector box jitter destroys speed/acceleration
statistics, while turning and path wander survive). If that is real, the best subset
differs between the two and a method tuned on GT trajectories optimises the wrong thing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_ceiling import FEATURES  # noqa: E402
from feature_selection import greedy_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
LOGS = ROOT / "logs"

# units nested inside the original sweep; every other unit's log is logs/fsn_<unit>.log
NESTED_LOG = {"ceiling": "fs_ceiling_nested.log",
              "bytetrack_longest": "fs_bytetrack_longest_nested.log"}
NESTED_RE = re.compile(r"(random forest|voting rule)\s+nested ([\d.]+) vs best-of-all ([\d.]+)")


def unit_order(name: str) -> tuple:
    """Ceiling first, then trackers alphabetically, oracle after longest."""
    return (0, "") if name == "ceiling" else (1, name)


def nested_rows(unit: str, full: pd.Series) -> list[dict]:
    """The two nested result lines (RF, voting) from a unit's log, against its all-7 row."""
    log = LOGS / NESTED_LOG.get(unit, f"fsn_{unit}.log")
    if not log.exists():
        return []
    lines = log.read_text(errors="replace").splitlines()
    rows = []
    for i, line in enumerate(lines):
        m = NESTED_RE.search(line)
        if not m:
            continue
        rule = "rf" if m.group(1) == "random forest" else "vote"
        picked = ast.literal_eval(lines[i + 1].split("folds picked:", 1)[1].strip())
        nested, best = float(m.group(2)), float(m.group(3))
        all7 = full.rf_macro_f1 if rule == "rf" else full.vote_macro_f1
        rows.append({
            "unit": unit, "rule": rule,
            "best_of_127": best, "nested": nested,
            "selection_bias": round(nested - best, 3),
            "all7": all7, "honest_gain": round(nested - all7, 4),
            **{f"folds_{f}": sum(f in p.split("+") for p in picked) for f in FEATURES},
            "picked": " | ".join(picked),
        })
    return rows


def main() -> int:
    paths = sorted(RES.glob("feature_selection_*.csv"))
    paths = [p for p in paths
             if not p.stem.endswith(("_summary", "_solo", "_entry", "_nested"))]
    if not paths:
        print("no feature_selection_<unit>.csv found -- run the sweep first")
        return 1

    units = {p.stem.replace("feature_selection_", ""): pd.read_csv(p) for p in paths}
    names = sorted(units, key=unit_order)

    summary, solo_rows, entry_rows, nest_rows = [], [], [], []
    for u in names:
        res = units[u]
        full = res[res.features == "+".join(FEATURES)].iloc[0]
        best_rf = res.loc[res.rf_macro_f1.idxmax()]
        best_v = res.loc[res.vote_macro_f1.idxmax()]
        solo = res[res.k == 1]
        best_solo = solo.loc[solo.rf_macro_f1.idxmax()]

        summary.append({
            "unit": u,
            "best_solo_feature": best_solo.features,
            "best_solo_rf_f1": best_solo.rf_macro_f1,
            "all7_rf_f1": full.rf_macro_f1,
            "all7_rf_acc": full.rf_accuracy,
            "best_subset_rf": best_rf.features,
            "best_subset_rf_k": int(best_rf.k),
            "best_subset_rf_f1": best_rf.rf_macro_f1,
            "best_subset_rf_acc": best_rf.rf_accuracy,
            "rf_gain_over_all7": round(best_rf.rf_macro_f1 - full.rf_macro_f1, 4),
            "all7_vote_f1": full.vote_macro_f1,
            "best_subset_vote": best_v.features,
            "best_subset_vote_k": int(best_v.k),
            "best_subset_vote_f1": best_v.vote_macro_f1,
            "vote_gain_over_all7": round(best_v.vote_macro_f1 - full.vote_macro_f1, 4),
        })

        for _, r in solo.iterrows():
            solo_rows.append({"unit": u, "feature": r.features,
                              "rf_macro_f1": r.rf_macro_f1, "rf_accuracy": r.rf_accuracy,
                              "vote_macro_f1": r.vote_macro_f1,
                              "vote_accuracy": r.vote_accuracy})

        order = {s["added"]: s["n"] for s in greedy_path(res, "rf_macro_f1")}
        entry_rows.append({"unit": u, **{f: order.get(f) for f in FEATURES}})
        nest_rows += nested_rows(u, full)

    sm = pd.DataFrame(summary)
    so = pd.DataFrame(solo_rows)
    en = pd.DataFrame(entry_rows)
    sm.to_csv(RES / "feature_selection_summary.csv", index=False)
    so.to_csv(RES / "feature_selection_solo.csv", index=False)
    en.to_csv(RES / "feature_selection_entry.csv", index=False)
    ne = pd.DataFrame(nest_rows)
    if len(ne):
        ne.to_csv(RES / "feature_selection_nested.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("=" * 110)
    print("1) PER UNIT: baseline -> best single feature -> best combination (random forest)")
    print("=" * 110)
    print(f"{'unit':24s}{'best solo':14s}{'solo F1':>9}{'all7 F1':>9}"
          f"{'best subset':>46s}{'k':>3}{'F1':>8}{'vs all7':>9}")
    for _, r in sm.iterrows():
        print(f"{r.unit:24s}{r.best_solo_feature:14s}{r.best_solo_rf_f1:>9.3f}"
              f"{r.all7_rf_f1:>9.3f}{r.best_subset_rf:>46s}{r.best_subset_rf_k:>3d}"
              f"{r.best_subset_rf_f1:>8.3f}{r.rf_gain_over_all7:>+9.3f}")

    print("\n" + "=" * 110)
    print("2) EACH FEATURE ALONE (random forest macro F1) -- add one feature to the bare pipeline")
    print("=" * 110)
    piv = so.pivot(index="unit", columns="feature", values="rf_macro_f1")
    piv = piv.reindex(index=[u for u in names], columns=FEATURES)
    print(piv.round(3).to_string())

    print("\n" + "=" * 110)
    print("3) GREEDY ENTRY STEP (1 = picked first; blank = never improved the set)")
    print("=" * 110)
    en2 = en.set_index("unit").reindex([u for u in names])[FEATURES]
    print(en2.to_string(na_rep="-"))

    print("\n" + "=" * 110)
    print("4) THE VOTING RULE (Aruna's method), same layout as 1")
    print("=" * 110)
    print(f"{'unit':24s}{'all7 F1':>9}{'best subset':>46s}{'k':>3}{'F1':>8}{'vs all7':>9}")
    for _, r in sm.iterrows():
        print(f"{r.unit:24s}{r.all7_vote_f1:>9.3f}{r.best_subset_vote:>46s}"
              f"{r.best_subset_vote_k:>3d}{r.best_subset_vote_f1:>8.3f}"
              f"{r.vote_gain_over_all7:>+9.3f}")

    if len(ne):
        print("\n" + "=" * 110)
        print("5) HONESTY CHECK -- nested (selection redone inside each outer fold) vs all 7 features")
        print("=" * 110)
        print(f"{'unit':24s}{'rule':>5}{'best-127':>10}{'nested':>8}{'bias':>8}{'all 7':>8}"
              f"{'honest gain':>13}{'tortuosity':>12}{'accel_std':>11}")
        for _, r in ne.iterrows():
            print(f"{r.unit:24s}{r.rule:>5}{r.best_of_127:>10.3f}{r.nested:>8.3f}"
                  f"{r.selection_bias:>+8.3f}{r.all7:>8.3f}{r.honest_gain:>+13.3f}"
                  f"{r.folds_tortuosity:>10d}/5{r.folds_accel_std:>9d}/5")
        real = ne[ne.unit != "ceiling"]
        for rule in ("rf", "vote"):
            g = real[real.rule == rule].honest_gain
            print(f"   real-track units, {rule:4s}: honest gain > 0 on {(g > 0).sum()}/{len(g)}, "
                  f"< 0 on {(g < 0).sum()}/{len(g)}, > +0.05 on {(g > 0.05).sum()}/{len(g)} | "
                  f"median {g.median():+.3f}, range {g.min():+.3f} .. {g.max():+.3f}")

    print(f"\n-> {RES}/feature_selection_summary.csv, _solo.csv, _entry.csv"
          f"{', _nested.csv' if len(ne) else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
