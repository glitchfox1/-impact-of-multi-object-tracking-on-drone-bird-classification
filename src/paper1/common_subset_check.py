#!/usr/bin/env python3
"""Robustness check: does the tracker ranking survive scoring every tracker on the SAME clips?

results/tracker_classification.csv scores each tracker on the clips it produced a track for,
so the clip sets differ: drones are 35/35 everywhere, but 0-4 birds drop out under
policy=longest and 3-8 under oracle, and WHICH birds drop depends on the tracker.
strongsort and hybridsort keep V_BIRD_016 / V_BIRD_021 -- clips whose best detection is
below the 0.45 track-start threshold -- while the other seven lose them. Cross-tracker
differences are therefore partly differences in which clips were scored.

This script re-scores every tracker/policy, and the perfect-track ceiling, on the clips
COMMON to all trackers under that policy, with the identical classifier suite
(tracker_features.evaluate). It first re-scores each unit on its FULL clip set and requires
that to reproduce the published table to 4 dp, so any difference in the common-set numbers
is the clip set and nothing else.

Decision af9dd5c9: if every tracker's RF macro F1 moves by <= 0.05 (the CV noise band at
n~66-70) and the top unit is unchanged, the per-tracker table stands and is disclosed;
otherwise the full common-subset re-run is triggered.

Writes results/tracker_classification_common.csv and results/tracker_dropped_clips.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker_features import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
CLASSES = ("BIRD", "DRONE")
CONF_HIGH = 0.45     # run_trackers.CONF_HIGH, the shared track-start threshold
NOISE = 0.05         # CV noise band at n~66-70


def det_conf(video: str) -> tuple[float, int]:
    """(max detection confidence, detections >= CONF_HIGH) from the cached detections."""
    f = next(RES.glob(f"dets/fold*/{video}.npy"))
    a = np.load(f)
    return (float(a[:, 5].max()), int((a[:, 5] >= CONF_HIGH).sum())) if len(a) else (0.0, 0)


def score(df: pd.DataFrame, label: str) -> dict:
    """evaluate() on one clip set, minus its own 'unit' label (the caller keys the row)."""
    rows: list = []
    evaluate(df.reset_index(drop=True), label, rows)
    rows[0].pop("unit")
    return rows[0]


def main() -> int:
    ceiling = pd.read_csv(RES / "ceiling_features_track.csv")
    ceiling = ceiling[ceiling.cls.isin(CLASSES)]
    all_clips = set(ceiling.video)
    published = pd.read_csv(RES / "tracker_classification.csv").set_index("unit")

    feats = {}
    for f in sorted(RES.glob("tracker_features_*.csv")):
        d = pd.read_csv(f)
        feats[d.tracker.iloc[0]] = d[d.cls.isin(CLASSES)]

    out, dropped = [], []
    for policy in ("longest", "oracle"):
        per = {t: d[d.policy == policy] for t, d in feats.items()}
        # ... and that have a usable GT track (ADR-006), so the ceiling row uses identical clips
        common = set.intersection(*(set(d.video) for d in per.values())) & all_clips
        cc = ceiling[ceiling.video.isin(common)]
        print(f"\n##### policy={policy}: common set {len(common)} clips "
              f"({dict(cc.cls.value_counts())}), dropped from it: "
              f"{sorted(all_clips - common)}")

        c = score(cc, f"ceiling/common-{policy}")
        out.append({"tracker": "ceiling", "policy": policy, "set": "common", **c})

        for t, d in per.items():
            unit = f"{t}/{policy}"
            full = score(d, f"{unit} full")
            pub = published.loc[unit]
            for k in ("rf_macro_f1", "fitted_vote_macro_f1", "thesis_macro_f1"):
                if abs(full[k] - pub[k]) > 1e-4:
                    raise SystemExit(f"REPRODUCTION FAILED {unit} {k}: "
                                     f"{full[k]} vs published {pub[k]}")
            com = score(d[d.video.isin(common)], f"{unit} common")
            out.append({"tracker": t, "policy": policy, "set": "full", **full})
            out.append({"tracker": t, "policy": policy, "set": "common", **com})

            for v in sorted(all_clips - set(d.video)):
                mx, n_hi = det_conf(v)
                dropped.append({"tracker": t, "policy": policy, "video": v,
                                "cls": ceiling.set_index("video").cls[v],
                                "det_conf_max": round(mx, 3),
                                f"dets_ge_{CONF_HIGH}": n_hi})
    print("\nreproduction check PASSED: every full-set score matches "
          "tracker_classification.csv to 4 dp")

    res = pd.DataFrame(out)
    res.to_csv(RES / "tracker_classification_common.csv", index=False)
    pd.DataFrame(dropped).to_csv(RES / "tracker_dropped_clips.csv", index=False)

    # ---- verdict against decision af9dd5c9
    verdict_ok = True
    for policy in ("longest", "oracle"):
        p = res[(res.policy == policy) & (res.tracker != "ceiling")]
        w = p.pivot(index="tracker", columns="set",
                    values=["rf_macro_f1", "fitted_vote_macro_f1"])
        w.columns = [f"{a}_{b}" for a, b in w.columns]
        w["rf_delta"] = w.rf_macro_f1_common - w.rf_macro_f1_full
        w["vote_delta"] = w.fitted_vote_macro_f1_common - w.fitted_vote_macro_f1_full
        w["rank_full"] = w.rf_macro_f1_full.rank(ascending=False).astype(int)
        w["rank_common"] = w.rf_macro_f1_common.rank(ascending=False).astype(int)
        w = w.sort_values("rf_macro_f1_full", ascending=False)
        print(f"\n=== policy={policy}: RF / fitted vote, full vs common ===")
        print(w.round(4).to_string())
        top_full, top_com = w.rf_macro_f1_full.idxmax(), w.rf_macro_f1_common.idxmax()
        big = w.index[w.rf_delta.abs() > NOISE].tolist()
        print(f"   top RF unit: full {top_full} | common {top_com}"
              f"{'  (UNCHANGED)' if top_full == top_com else '  (CHANGED)'}")
        print(f"   RF moves > {NOISE}: {big or 'none'} | "
              f"max |RF delta| {w.rf_delta.abs().max():.4f}")
        if policy == "longest" and (big or top_full != top_com):
            verdict_ok = False
    print(f"\nVERDICT (decision af9dd5c9, policy=longest): "
          f"{'per-tracker table STANDS' if verdict_ok else 'TRIGGERS the full common-subset re-run'}")
    print(f"-> {RES / 'tracker_classification_common.csv'}\n-> {RES / 'tracker_dropped_clips.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
