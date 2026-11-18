#!/usr/bin/env python3
"""Train the class-agnostic 'flying object' detector, one per cross-validation fold.

Why one per fold: the classification experiment reports 5-fold CV over the same clips.
A single detector trained on everything would have seen every test clip, so the tracks
it produces on the test fold would be unrealistically clean and the reported accuracy
would be inflated. Fold k's detector never sees fold k's clips.

Why an inner validation split (ADR-007, 2026-09-25): until then the held-out fold was also
the YOLO `val` set, so it chose the best epoch and triggered early stopping -- the test fold
steered model selection. Now fold k trains on fold<k>_itrain, selects on fold<k>_ival (whole
training clips set aside, see export_yolo_frames.py), and fold k itself is touched only by
the final evaluation on fold<k>_test.

Why one class: the detector must not know bird from drone. That call is what the 7
track features are being tested on.

Targets here are tiny (median box ~30x25 px in 640x512), so mosaic is kept on for
scale variety but the model is small and the image is used at native resolution.

Writes <project>/fold<k>/weights/best.pt and the per-fold summary (--summary): held-out
test metrics under the original keys (precision, recall, map50, map50_95), inner-validation
metrics under ival_*. Refuses to overwrite existing weights unless --overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "svan_det"
OUT = ROOT / "results" / "detector"


def write_yaml(fold: int) -> Path:
    for split in ("itrain", "ival", "test"):
        if not (DATA / "folds" / f"fold{fold}_{split}.txt").exists():
            raise SystemExit(f"missing fold{fold}_{split}.txt: run export_yolo_frames.py --labels-only")
    y = DATA / f"fold{fold}.yaml"
    y.write_text(
        f"path: {DATA.as_posix()}\n"
        f"train: folds/fold{fold}_itrain.txt\n"
        f"val: folds/fold{fold}_ival.txt\n"
        f"test: folds/fold{fold}_test.txt\n"
        "names:\n  0: flying_object\n"
    )
    return y


def metrics(m, prefix: str = "") -> dict:
    return {f"{prefix}precision": float(m.box.mp), f"{prefix}recall": float(m.box.mr),
            f"{prefix}map50": float(m.box.map50), f"{prefix}map50_95": float(m.box.map)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0", help="comma list, e.g. 0,1,2,3,4")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)      # Windows: 2 max (CLAUDE.md)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of itrain used (tests only)")
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    ap.add_argument("--summary", type=Path, default=OUT / "detector_summary.json")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing weights")
    args = ap.parse_args()
    # Ultralytics puts a RELATIVE project under its own runs_dir (the 2026-09-25 test run with
    # --project runs/detect_test landed in runs/detect/runs/detect_test), so the weights path
    # computed below would miss it. Absolute paths mean the same thing to both.
    args.project, args.summary = args.project.resolve(), args.summary.resolve()

    from ultralytics import YOLO

    folds = [int(f) for f in args.folds.split(",")]
    existing = [args.project / f"fold{k}" / "weights" / "best.pt" for k in folds]
    existing = [p for p in existing if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit("refusing to overwrite existing detector weights (archive them, or pass "
                         "--overwrite):\n  " + "\n  ".join(map(str, existing)))

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        yaml_path = write_yaml(fold)
        print(f"\n=== fold {fold} -> {yaml_path} ===", flush=True)
        model = YOLO(args.model)
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            fraction=args.fraction,
            project=str(args.project),
            name=f"fold{fold}",
            exist_ok=True,
            device=0,
            seed=0,
            val=True,
            plots=False,
            deterministic=False,
        )
        run_dir = args.project / f"fold{fold}"
        weights = run_dir / "weights" / "best.pt"
        best = YOLO(str(weights))                  # evaluate the selected epoch explicitly
        kw = dict(data=str(yaml_path), imgsz=args.imgsz, batch=args.batch,
                  workers=args.workers, device=0, plots=False, verbose=False)
        m_ival = best.val(split="val", project=str(run_dir), name="eval_ival", exist_ok=True, **kw)
        m_test = best.val(split="test", project=str(run_dir), name="eval_test", exist_ok=True, **kw)
        with (run_dir / "results.csv").open(newline="") as fh:
            hist = list(csv.DictReader(fh))
        # Ultralytics 8.4 picks best.pt by fitness = mAP50-95 alone (DetMetrics.fitness,
        # weights 0/0/0/1; older versions used 0.1*mAP50 + 0.9*mAP50-95). The stripped
        # checkpoint stores epoch -1, so the best epoch is read back from results.csv.
        fit = [float(r["metrics/mAP50-95(B)"]) for r in hist]
        entry = {**metrics(m_test), **metrics(m_ival, "ival_"),
                 "epochs_run": len(hist), "best_epoch": int(fit.index(max(fit))) + 1,
                 "selection": "inner validation (fold%d_ival), ADR-007" % fold,
                 "weights": str(weights)}
        prev = json.loads(args.summary.read_text()) if args.summary.exists() else {}
        prev[str(fold)] = entry                    # per fold, so a crash keeps finished folds
        args.summary.write_text(json.dumps(prev, indent=2))
        print(f"fold {fold}: TEST P {entry['precision']:.3f}  R {entry['recall']:.3f}  "
              f"mAP50 {entry['map50']:.3f}  mAP50-95 {entry['map50_95']:.3f}  |  "
              f"ival mAP50 {entry['ival_map50']:.3f}  |  best epoch "
              f"{entry['best_epoch']}/{entry['epochs_run']}", flush=True)

    print(f"\nwrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
