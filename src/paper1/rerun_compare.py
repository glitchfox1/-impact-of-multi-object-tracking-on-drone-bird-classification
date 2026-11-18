#!/usr/bin/env python3
"""Before/after tables for the 2026-09-26 full rerun (ADR-007 detector retrain + ADR-008 tracker fix).

v1 = the "before" archive (--v1; default results/archive_2026-09-26_pre_rerun, the manuscript as of 2026-09-24),
v2 = results/ after scripts/run_full_rerun.sh. For the ADR-009 rerun: --v1 results/archive_2026-09-26b_pre_48clips
--v1-weights runs/detect/_v2_2026-09-26 (so "v1" columns hold v2 and "v2" columns hold v3).

--eval-v1   ALSO evaluates the five v1 detectors (runs/detect/_v1_2026-09-18/fold<k>) on the NEW held-out
            lists fold<k>_test. Needed for a fair detector comparison: v1's summary was measured on
            fold<k>_val, which still carried the drifted labels and the unlabelled frames of the corrected
            clips, so v1-summary vs v2-summary compares two different test sets. GPU; run only when no
            training is running (8 GB card).
            -> results/detector/v1_on_new_test_lists.json

Writes results/rerun_comparison_<table>.csv and prints the headline differences.

    python src/paper1/rerun_compare.py [--eval-v1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
V1 = RES / "archive_2026-09-26_pre_rerun"
V1_WEIGHTS = ROOT / "runs" / "detect" / "_v1_2026-09-18"
DATA = ROOT / "datasets" / "svan_det"
KEYS = ("precision", "recall", "map50", "map50_95")


def eval_v1_on_new_lists(imgsz: int = 640, batch: int = 16) -> dict:
    from ultralytics import YOLO
    out = {}
    for k in range(5):
        best = YOLO(str(V1_WEIGHTS / f"fold{k}" / "weights" / "best.pt"))
        m = best.val(data=str(DATA / f"fold{k}.yaml"), split="test", imgsz=imgsz, batch=batch, workers=2,
                     device=0, plots=False, verbose=False,
                     project=str(V1_WEIGHTS / f"fold{k}"), name="eval_new_test", exist_ok=True)
        out[str(k)] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                       "map50": float(m.box.map50), "map50_95": float(m.box.map)}
        print(f"v1 fold {k} on fold{k}_test: " + "  ".join(f"{c} {out[str(k)][c]:.3f}" for c in KEYS), flush=True)
    p = RES / "detector" / "v1_on_new_test_lists.json"
    p.write_text(json.dumps(out, indent=2))
    return out


def detectors(v1_new: dict | None) -> pd.DataFrame:
    v1 = json.loads((V1 / "detector" / "detector_summary.json").read_text())
    v2 = json.loads((RES / "detector" / "detector_summary.json").read_text())
    if v1_new is None and (RES / "detector" / "v1_on_new_test_lists.json").exists():
        v1_new = json.loads((RES / "detector" / "v1_on_new_test_lists.json").read_text())
    rows = []
    for k in sorted(v2, key=int):
        r = {"fold": int(k), "v2_best_epoch": v2[k].get("best_epoch"), "v2_epochs_run": v2[k].get("epochs_run"),
             "v2_ival_map50": v2[k].get("ival_map50")}
        for c in KEYS:
            r[f"v1_old_lists_{c}"] = v1.get(k, {}).get(c)
            if v1_new:
                r[f"v1_new_lists_{c}"] = v1_new.get(k, {}).get(c)
            r[f"v2_{c}"] = v2[k][c]
        rows.append(r)
    return pd.DataFrame(rows)


def quality() -> pd.DataFrame:
    """Per tracker and class: tracks per clip, ID switches, oracle coverage, detection recall."""
    def agg(p: Path, tag: str) -> pd.DataFrame:
        q = pd.read_csv(p)
        g = q.groupby(["tracker", "cls"]).agg(clips=("video", "nunique"), tracks_per_clip=("n_tracks", "mean"),
                                              id_switches=("id_switches", "sum"),
                                              coverage=("oracle_coverage", "mean"), det_recall=("det_recall", "mean"))
        return g.add_prefix(f"{tag}_")
    return agg(V1 / "tracker_quality.csv", "v1").join(agg(RES / "tracker_quality.csv", "v2"), how="outer").reset_index()


def classification() -> pd.DataFrame:
    cols = ["n", "fitted_vote_macro_f1", "rf_macro_f1"]
    a = pd.read_csv(V1 / "tracker_classification.csv").set_index("unit")[cols].add_prefix("v1_")
    b = pd.read_csv(RES / "tracker_classification.csv").set_index("unit")[cols].add_prefix("v2_")
    d = a.join(b, how="outer")
    d["rf_delta"] = d.v2_rf_macro_f1 - d.v1_rf_macro_f1
    return d.reset_index()


def hota() -> pd.DataFrame:
    cols = ["HOTA", "IDF1", "MOTA", "IDSW"]
    a = pd.read_csv(V1 / "tracking_metrics_birddrone.csv").set_index(["tracker", "group"])[cols].add_prefix("v1_")
    b = pd.read_csv(RES / "tracking_metrics_birddrone.csv").set_index(["tracker", "group"])[cols].add_prefix("v2_")
    d = a.join(b, how="outer")
    d["HOTA_delta"] = d.v2_HOTA - d.v1_HOTA
    return d.reset_index()


def main() -> int:
    global V1, V1_WEIGHTS          # --v1 / --v1-weights rebind the module defaults
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-v1", action="store_true")
    ap.add_argument("--v1", type=Path, default=V1, help="archive holding the before results")
    ap.add_argument("--v1-weights", type=Path, default=V1_WEIGHTS, help="runs dir holding the before detectors")
    args = ap.parse_args()
    V1, V1_WEIGHTS = args.v1.resolve(), args.v1_weights.resolve()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    v1_new = eval_v1_on_new_lists() if args.eval_v1 else None
    tables = {"detector": detectors(v1_new), "quality": quality(),
              "classification": classification(), "hota": hota()}
    for name, df in tables.items():
        df.to_csv(RES / f"rerun_comparison_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
