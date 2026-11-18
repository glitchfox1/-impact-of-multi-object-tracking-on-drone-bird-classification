#!/usr/bin/env python3
"""Build a balanced Svanström subset (default 35 clips per class) as MOT-format GT.

Balancing counts alone is not enough. The classes differ in how far away the target
is, and apparent size is exactly what a bird/drone classifier could cheat on:
usable visible clips per class x distance bin are

    CLOSE  DISTANT  MEDIUM
    9      19       20      BIRD
    21     25       66      DRONE
    17     25       17      AIRPLANE
    27     10       23      HELICOPTER

so an unmatched sample would hand the classifier "DISTANT => helicopter is unlikely".
This script therefore draws the SAME distance composition for every class, capped by
the scarcest class in each bin (9 / 10 / 16 by default = 35 clips).

Each clip carries one target, so the ground truth is one track per sequence.

Outputs
    <out>/<video>/gt/gt.txt      frame,id,x,y,w,h,conf,class_id,visibility   (1-based)
    <out>/<video>/seqinfo.ini
    <out>/_manifest.csv          every usable clip, selected or not, with fold + provenance
    <out>/_summary.txt           counts actually achieved

Provenance warning carried into the manifest: BIRD and DRONE clips were filmed by the
dataset authors, but 11 AIRPLANE and 38 HELICOPTER visible clips are YouTube footage
(column "DRONE TYPE / INTERNET SOURCE"). Mixing those into a 4-class experiment means
the extra classes differ by camera as well as by object.

Label corrections (ADR-006): some visible bird labels drift off the bird. The hand-reviewed
frame ranges in gt/svanstrom_corrections.csv are applied here, so a rebuild reproduces the
fix instead of silently undoing it. Only `drop` exists: a dropped frame is left unlabelled,
never redrawn. boxes_written in the manifest counts what is written AFTER corrections.

--extend (ADR-009, 2026-09-26): keep every clip already selected in <out>/_manifest.csv WITH
its fold, and add clips only where a --target CLASS:BIN=N asks for more. Used to grow BIRD and
DRONE from 35 to 48 per class (every usable visible bird: CLOSE 9 / DISTANT 19 / MEDIUM 20)
without redrawing the audited clips or reshuffling the folds. A new clip is drawn (seeded) from
the unselected usable pool of its class x bin and joins the fold holding the fewest clips of
that class x bin (ties: the smaller fold overall, then the lower index).

    python src/paper1/build_svanstrom_subset.py --extend \\
        --target BIRD:DISTANT=19 --target BIRD:MEDIUM=20 \\
        --target DRONE:DISTANT=19 --target DRONE:MEDIUM=20
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from mcos_decoder import load_groundtruth

CLASS_ID = {"DRONE": 1, "BIRD": 2, "AIRPLANE": 3, "HELICOPTER": 4}
BINS = ["CLOSE", "DISTANT", "MEDIUM"]
MIN_LABELLED = 50          # clips with fewer labelled frames are unusable
FOLDS = 5


def decoded_frames(video: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def load_corrections(path: Path) -> dict[str, list[tuple[int, int]]]:
    """video -> [(first_frame, last_frame), ...] ranges to drop (1-based, inclusive)."""
    if not path.exists():
        return {}
    corr = pd.read_csv(path, comment="#")
    bad = set(corr.action) - {"drop"}
    if bad:
        raise SystemExit(f"{path}: unknown action(s) {bad}; only 'drop' is supported")
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in corr.itertuples():
        out[r.video].append((int(r.first_frame), int(r.last_frame)))
    return dict(out)


def probe_stream(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True).stdout.strip().split(",")
    w, h, rate = int(out[0]), int(out[1]), out[2]
    num, _, den = rate.partition("/")
    return w, h, round(float(num) / float(den or 1), 3)


def fresh_selection(args, usable: list[dict]) -> tuple[list[dict], dict[str, int], dict]:
    """The original draw: same distance composition for every class, 5 stratified folds."""
    # how many per bin: the scarcest class decides, so every class gets the same shape
    per_bin_avail = {b: {cls: sum(1 for c in usable if c["cls"] == cls and c["bin"] == b)
                         for cls in args.classes} for b in BINS}
    caps = {b: min(per_bin_avail[b].values()) for b in BINS}
    total_cap = sum(caps.values())
    if total_cap < args.per_class:
        print(f"  NOTE: matched sampling allows at most {total_cap} clips per class "
              f"({caps}); asked for {args.per_class}")
    target = {}
    left = min(args.per_class, total_cap)
    for b in sorted(BINS, key=lambda b: caps[b]):        # fill scarcest bins first
        share = min(caps[b], -(-left // (len(BINS) - len(target))))
        target[b] = share
        left -= share
    print(f"  per-class distance composition: {target} (available caps {caps})")

    rng = random.Random(args.seed)
    selected = []
    for cls in args.classes:
        for b in BINS:
            pool = sorted((c for c in usable if c["cls"] == cls and c["bin"] == b),
                          key=lambda c: c["video"])
            selected += rng.sample(pool, target[b])

    # 5 folds, stratified inside each class x bin so folds keep the same shape
    fold_of, counter = {}, defaultdict(int)
    for c in sorted(selected, key=lambda c: (c["cls"], c["bin"], c["video"])):
        key = (c["cls"], c["bin"])
        fold_of[c["video"]] = counter[key] % FOLDS
        counter[key] += 1
    return selected, fold_of, target


def extend_selection(args, clips: list[dict], usable: list[dict]) -> tuple[list[dict], dict[str, int], dict]:
    """Keep the manifest's selected clips and folds; add clips only where --target asks."""
    man = args.out / "_manifest.csv"
    if not man.exists():
        raise SystemExit(f"--extend needs an existing {man}")
    prev = pd.read_csv(man)
    prev = prev[prev.selected == 1]
    fold_of = {r.video: int(r.fold) for r in prev.itertuples()}
    by_name = {c["video"]: c for c in clips}
    missing = set(fold_of) - set(by_name)
    if missing:
        raise SystemExit(f"manifest clips no longer in the dataset: {sorted(missing)}")
    selected = [by_name[v] for v in sorted(fold_of)]

    targets: dict[tuple[str, str], int] = {}
    for t in args.target:
        cell, _, n = t.partition("=")
        cls, _, b = cell.upper().partition(":")
        if cls not in CLASS_ID or b not in BINS or not n.isdigit():
            raise SystemExit(f"bad --target {t!r}: want CLASS:BIN=N, BIN in {BINS}")
        targets[(cls, b)] = int(n)

    rng = random.Random(args.seed)
    for (cls, b), want in sorted(targets.items()):
        have = [c for c in selected if c["cls"] == cls and c["bin"] == b]
        if want < len(have):
            raise SystemExit(f"--target {cls}:{b}={want} is below the {len(have)} clips already "
                             "selected; --extend never removes clips")
        pool = sorted((c for c in usable if c["cls"] == cls and c["bin"] == b
                       and c["video"] not in fold_of), key=lambda c: c["video"])
        need = want - len(have)
        if need > len(pool):
            raise SystemExit(f"{cls}:{b}: need {need} more clips, only {len(pool)} usable left")
        for c in sorted(rng.sample(pool, need), key=lambda c: c["video"]):
            in_cell = [sum(1 for s in selected if s["cls"] == cls and s["bin"] == b
                           and fold_of[s["video"]] == k) for k in range(FOLDS)]
            size = [sum(1 for f in fold_of.values() if f == k) for k in range(FOLDS)]
            fold_of[c["video"]] = min(range(FOLDS), key=lambda k: (in_cell[k], size[k], k))
            selected.append(c)
        print(f"  {cls:10s} {b:7s} {len(have)} -> {want}")

    target = {"extend": {f"{c}:{b}": n for (c, b), n in sorted(targets.items())}}
    return selected, fold_of, target


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=root / "svanstrom_dataset" / "Data" / "Video_V")
    ap.add_argument("--xlsx", type=Path,
                    default=root / "svanstrom_dataset" / "Data" / "Video_dataset_description.xlsx")
    ap.add_argument("--out", type=Path, default=root / "gt" / "svanstrom_visible")
    ap.add_argument("--corrections", type=Path, default=root / "gt" / "svanstrom_corrections.csv")
    ap.add_argument("--per-class", type=int, default=35)
    ap.add_argument("--classes", nargs="+", default=list(CLASS_ID))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extend", action="store_true",
                    help="keep the clips and folds already selected in <out>/_manifest.csv; "
                         "add clips only to reach --target")
    ap.add_argument("--target", action="append", default=[], metavar="CLASS:BIN=N",
                    help="with --extend: clips wanted in one class x distance bin, e.g. BIRD:DISTANT=19")
    args = ap.parse_args()

    meta = pd.ExcelFile(args.xlsx).parse("Blad1")
    meta.columns = [c.strip() for c in meta.columns]
    meta = meta[meta.SENSOR.str.upper().str.startswith("V")]

    clips = []
    for _, r in meta.iterrows():
        cls = r.CLASS.upper()
        if cls not in args.classes:
            continue
        name = f"V_{cls}_{int(r.NUMBER):03d}"
        lab, vid = args.data / f"{name}_LABELS.mat", args.data / f"{name}.mp4"
        if not lab.exists() or not vid.exists():
            print(f"  missing files for {name}")
            continue
        boxes = load_groundtruth(str(lab))
        n_lab = sum(1 for b in boxes if b is not None)
        src = r["DRONE TYPE / INTERNET SOURCE"]
        clips.append({
            "video": name, "cls": cls, "bin": str(r["DISTANCE  BIN"]).strip(),
            "labelled": n_lab, "boxes": boxes, "video_path": vid,
            "provenance": "internet" if isinstance(src, str) and src.startswith("http") else "authors",
            "usable": n_lab >= MIN_LABELLED,
        })

    usable = [c for c in clips if c["usable"]]
    if args.extend:
        selected, fold_of, target = extend_selection(args, clips, usable)
    else:
        selected, fold_of, target = fresh_selection(args, usable)
    chosen = {c["video"] for c in selected}

    args.out.mkdir(parents=True, exist_ok=True)
    summary = defaultdict(lambda: [0, 0])     # cls -> [clips, boxes]
    dropped_past_end = 0
    corrections = load_corrections(args.corrections)
    unknown = set(corrections) - chosen
    if unknown:
        raise SystemExit(f"corrections name clips that are not selected: {sorted(unknown)}")
    dropped_by_correction: dict[str, int] = defaultdict(int)

    for c in selected:
        w, h, fps = probe_stream(c["video_path"])
        n_frames = decoded_frames(c["video_path"])
        seq = args.out / c["video"]
        (seq / "gt").mkdir(parents=True, exist_ok=True)
        lines = []
        for idx, box in enumerate(c["boxes"], start=1):   # label index == 1-based frame
            if box is None:
                continue
            if idx > n_frames:                            # labels can outrun the video
                dropped_past_end += 1
                continue
            if any(a <= idx <= b for a, b in corrections.get(c["video"], ())):
                dropped_by_correction[c["video"]] += 1    # hand-reviewed drift (ADR-006)
                continue
            x, y, bw, bh = box
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            lines.append(f"{idx},1,{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1,{CLASS_ID[c['cls']]},1")
        (seq / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", newline="\n")
        (seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={c['video']}\nimWidth={w}\nimHeight={h}\n"
            f"seqLength={n_frames}\nframeRate={fps}\n", newline="\n")
        summary[c["cls"]][0] += 1
        summary[c["cls"]][1] += len(lines)
        c.update(width=w, height=h, fps=fps, frames=n_frames, written=len(lines))

    with (args.out / "_manifest.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "class", "class_id", "distance_bin", "provenance",
                     "labelled_frames", "usable", "selected", "fold",
                     "width", "height", "fps", "frames", "boxes_written"])
        for c in sorted(clips, key=lambda c: (c["cls"], c["video"])):
            sel = c["video"] in chosen
            wr.writerow([c["video"], c["cls"], CLASS_ID[c["cls"]], c["bin"], c["provenance"],
                         c["labelled"], int(c["usable"]), int(sel),
                         fold_of.get(c["video"], ""), c.get("width", ""), c.get("height", ""),
                         c.get("fps", ""), c.get("frames", ""), c.get("written", "")])

    text = [f"per-class target composition: {target}",
            f"dropped labels past the last video frame: {dropped_past_end}",
            f"dropped by gt corrections ({args.corrections.name}): "
            f"{dict(sorted(dropped_by_correction.items())) or 'none'}", ""]
    for cls in args.classes:
        clips_n, boxes_n = summary[cls]
        text.append(f"{cls:11s} {clips_n:3d} clips  {boxes_n:6d} boxes")
    (args.out / "_summary.txt").write_text("\n".join(text) + "\n", newline="\n")
    print("\n".join(text))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
