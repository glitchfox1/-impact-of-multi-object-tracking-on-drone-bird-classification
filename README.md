# Impact of Multi-Object Tracking on the Classification of Drones and Birds Using Trajectory Features

Code and figures for a controlled comparison of nine multi-object trackers in a
trajectory-based bird-versus-drone classifier. A single-class detector is trained per
cross-validation fold; its detections are cached once and replayed, byte for byte, into nine
[BoxMOT](https://github.com/mikel-brostrom/boxmot) trackers; seven interpretable trajectory
features are computed from every track and classified by a fitted voting rule and a random
forest; and the same feature code is run on ground-truth boxes to give a perfect-track reference.
The scripts also time every stage of the pipeline.

**This repository holds code and figures only.** It contains no video, no frames, no labels, no
detections, no tracks, no result tables and no manuscript.

## Contents

```
src/paper1/      the pipeline, statistics and figure scripts (19 Python files)
figures/         the figures, as PNG
LICENSE          GNU Affero General Public License v3.0
```

### Pipeline

Run from the repository root, in this order. Each script writes under `results/`, `datasets/` or
`runs/`, which it creates. `--help` lists the options of every script.

```
python src/paper1/build_svanstrom_subset.py        # clip selection and ground truth (see --help for the extension to 48 clips per class)
python src/paper1/export_yolo_frames.py            # frames and YOLO labels
python src/paper1/train_detector.py --folds 0,1,2,3,4
python src/paper1/run_trackers.py --folds 0,1,2,3,4
python src/paper1/operating_point_probe.py
python src/paper1/feature_ceiling.py               # perfect-track reference
python src/paper1/tracker_features.py              # real tracks
python src/paper1/common_subset_check.py
python src/paper1/paper_tables.py --settings
python src/paper1/paper_tables.py --benchmark      # cost of every stage
python src/paper1/paper_stats.py
python src/paper1/roc_auc.py
python src/paper1/reference_split.py
python src/paper1/feature_selection.py --tracker bytetrack --policy longest --nested   # once per tracker; omit --tracker for the reference
python src/paper1/feature_selection_report.py
python src/paper1/paper_figures.py
python src/paper1/qualitative_figures.py
```

| Script | What it does |
|---|---|
| `build_svanstrom_subset.py` | Selects a class-balanced, distance-matched set of clips and writes ground truth in MOT format. |
| `export_yolo_frames.py` | Decodes the selected clips into frames and YOLO labels. |
| `train_detector.py` | Trains the one-class "flying object" detector, one per fold, so a fold's detector never sees that fold's clips. |
| `run_trackers.py` | Caches detections once, then replays them into all nine trackers: ByteTrack, OC-SORT, SFSORT, BoostTrack, BoT-SORT, StrongSORT, Deep OC-SORT, Hybrid-SORT, OccluBoost. |
| `operating_point_probe.py` | Checks that every tracker honours the shared operating point (start 0.45, floor 0.10) on synthetic probe clips. |
| `feature_ceiling.py` | Perfect-track reference: the seven features on ground-truth boxes. |
| `tracker_features.py` | The same features on real tracks, scored for every tracker and track-selection policy. |
| `common_subset_check.py`, `reference_split.py` | Scores every tracker on the clips that all nine keep; splits the reference between original and added clips. |
| `feature_selection.py`, `feature_selection_report.py` | Feature subsets, with nested validation, and the table across trackers. |
| `paper_stats.py`, `roc_auc.py` | Clip-level bootstrap intervals, paired tests with Holm correction, ROC curves and AUC. |
| `paper_tables.py` | Tracker settings read back from the constructed trackers, and the cost of every stage. |
| `paper_figures.py`, `qualitative_figures.py` | The figures. |

`review_svanstrom_gt.py`, `preview_gt.py` and `rerun_compare.py` are helper scripts for checking
labels by eye and for comparing two runs.

## Data

The data are not included.

- **Svanström multi-sensor drone-detection dataset** (CC0): Svanström, F., Alonso-Fernandez, F.,
  Englund, C. A dataset for multi-sensor drone detection. *Data in Brief* 39, 107521 (2021).
  https://doi.org/10.1016/j.dib.2021.107521. The scripts expect the visible-light videos under
  `svanstrom_dataset/Data/Video_V/`.
- **Clip selection and ground truth.** Stages 1 and 2 also read `gt/svanstrom_visible/_manifest.csv`
  and `gt/svanstrom_corrections.csv`, which hold the selected clips and the hand-corrected
  ground-truth boxes. These are not part of this repository.
- **Drone-vs-Bird data** were used for one drone-only check under the organisers' data usage
  agreement and cannot be redistributed. Nothing from them is included here.

Per-clip predictions, statistics and the files in the second item are available from the
corresponding author on request.

## Requirements

Versions the results were produced with: Python 3.11, PyTorch 2.6.0 (CUDA 12.4), Ultralytics 8.4.155,
BoxMOT 25.0.0, OpenCV 4.14, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, matplotlib. `ffmpeg` is
needed by `preview_gt.py`. Everything ran on one laptop with an NVIDIA RTX 4060 (8 GB) and an Intel
i7-13620H.

BoxMOT 25.0.0 is not the API of earlier releases: its trackers live under `boxmot/trackers/`.

This repository has not been tested from a fresh clone. The pipeline expects the data layout
described above, so the scripts will not run until the data and the two ground-truth files are in
place.

## Figures

| File | Shows |
|---|---|
| `fig1_pipeline.png` | The pipeline: the tracker arm and the perfect-track reference arm. |
| `fig2_tracker_macro_f1.png` | Macro F1 per tracker for both classifiers, against the perfect-track reference. |
| `fig3_solo_feature_heatmap.png` | Macro F1 of each trajectory feature on its own, on perfect and real tracks. |
| `fig4_class_tracking_quality.png` | Median tracked-box recall and coverage for birds and drones, per tracker. |
| `fig5_tortuosity_confound.png` | Tortuosity and net displacement for birds and drones, on ground truth and on tracks. |
| `fig6_selection_bias.png` | The bias of quoting the best feature subset, against nested selection. |
| `fig7_roc.png` | ROC curves for both classifiers on the clips all trackers keep. |
| `fig_dataset_examples.png` | One example per class and distance bin. |
| `fig_failure_cases.png` | Tracking failure modes, with magnified insets. |
| `fig_tracking_strip.png` | Fragmentation of one bird track, ByteTrack above and StrongSORT below. |

The frames in these figures come from the Svanström dataset (CC0).

## Licence

AGPL-3.0, see `LICENSE`. The code builds on Ultralytics and BoxMOT, which are both AGPL-3.0.
