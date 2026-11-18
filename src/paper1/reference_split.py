#!/usr/bin/env python3
"""Perfect-track reference on the common clips, split into original and added clips.

The manuscript (sections 4.1, 5 and 6) quotes the reference's random-forest macro F1 on the
92 common clips (policy=longest) separately for the clips of the original 35-per-class subset
and for the 26 clips added by the ADR-009 extension, to show that the added clips are the
hard ones. This script derives that split from the saved out-of-fold predictions, so the
numbers have a script behind them instead of a one-off session calculation.

Original clips = BIRD/DRONE clips selected in gt/_pre_extend_2026-09-26/_manifest.csv.
The predictions are the ones the common-set comparison produced (retrained on those clips);
here they are only re-scored on the two subsets, with no retraining.

Writes results/reference_split.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
PRE_EXTEND = ROOT / "gt" / "_pre_extend_2026-09-26" / "_manifest.csv"
CLASSES = ("BIRD", "DRONE")


def main() -> int:
    oof = pd.read_csv(RES / "paper_oof_predictions.csv")
    ref = oof[(oof.tracker == "ceiling") & (oof.policy == "longest") & (oof.set == "common")]

    pre = pd.read_csv(PRE_EXTEND)
    original = set(pre[(pre.selected == 1) & pre["class"].isin(CLASSES)].video)
    ref = ref.assign(subset=ref.video.map(lambda v: "original" if v in original else "added"))

    rows = []
    for name, g in [("all", ref), *ref.groupby("subset")]:
        rows.append({"subset": name, "n": len(g),
                     "n_bird": int((g.cls == "BIRD").sum()), "n_drone": int((g.cls == "DRONE").sum()),
                     "rf_macro_f1": round(f1_score(g.cls, g.rf_pred, average="macro"), 4),
                     "vote_macro_f1": round(f1_score(g.cls, g.vote_pred, average="macro"), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "reference_split.csv", index=False)
    print(f"original clips in the pre-extension manifest: {len(original)}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
