#!/usr/bin/env python3
"""reflection_negatives.py — build a leak-free REFLECTION negative set for presence evaluation.

Context. The paper claims the segmenter is "reflection-robust" and reports presence AUC 0.794, but
that AUC was measured on 19 empty-tank negatives from the SAME cameras as the positives — the
reflection failure mode (Right_Left, where the camera sees a mirrored human/room and the CLIP
detector fires at p_visible=1.0) was never tested for the deployed thin768 model. The v3 negatives
model was tested on reflections; thin768 was not.

Leakage assertion (verified 2026-08-15, referee-required): thin768 trained on
`/dataset_seg_thin768` = 4,965 images, of which **0** are Right_Left. The reflection camera is
excluded by construction from `auto_segment.py` and from the human label set, so these frames are
leak-free for this model.

Sampling discipline (referee-required): <=`--per-clip` frames per clip, spread across >=`--min-videos`
distinct SOURCE VIDEOS (date/segment). 150 frames drawn from 5 clips is n=5, not n=150; downstream
statistics must cluster-bootstrap by video, and this file records the video of every frame so they can.

Verification discipline. Right_Left clips are NOT automatically octopus-free — the camera can see the
real animal as well as its reflection. This repo has a scar here: 166 of 232 mined "hard negatives"
turned out to contain the octopus (see docs/). So every sampled frame must be reviewed before it
is scored as a negative. This script only STAGES frames + a contact sheet; `verified` starts null and
is filled in by review. Nothing downstream may use a frame whose `verified` is not True.

Usage:
  venv/bin/python3 src/reflection_negatives.py --sample --n 150
  venv/bin/python3 src/reflection_negatives.py --contact-sheet     # montages for review
"""
import argparse, collections, json, random, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTDIR = REPO / "data" / "reflection_negatives"
INDEX = OUTDIR / "index.json"
CLIPS_ROOT = REPO / "src" / "octopus_clips_verified"      # 427 Right_Left clips live here


def source_video(p: Path):
    return f"{p.parent.parent.name}/{p.parent.name}"


def sample(n_total=150, per_clip=2, min_videos=20, seed=17):
    clips = sorted(CLIPS_ROOT.glob("*/*/Right_Left_*.mp4"))
    if not clips:
        sys.exit(f"no Right_Left clips under {CLIPS_ROOT}")
    by_vid = collections.defaultdict(list)
    for c in clips:
        by_vid[source_video(c)].append(c)
    vids = sorted(by_vid)
    rng = random.Random(seed)
    rng.shuffle(vids)
    print(f"pool: {len(clips)} Right_Left clips across {len(vids)} source videos")
    if len(vids) < min_videos:
        sys.exit(f"only {len(vids)} distinct videos (< --min-videos {min_videos}); "
                 "widen the pool before sampling — correlated frames are not independent samples")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, per_vid = [], collections.Counter()
    # round-robin over videos so no single recording dominates
    while len(rows) < n_total:
        progressed = False
        for v in vids:
            if len(rows) >= n_total:
                break
            pool = by_vid[v]
            k = per_vid[v]
            if k >= per_clip * len(pool):
                continue
            clip = pool[k % len(pool)]
            tmp = tempfile.mkdtemp()
            # deterministic frame: step through the clip so repeats sample different times
            t = 3.0 + 4.0 * (k // len(pool))
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(clip),
                            "-frames:v", "1", "-vf", "scale='min(1024,iw)':-2",
                            f"{tmp}/f.jpg"], check=False)
            f = Path(tmp) / "f.jpg"
            if f.exists() and cv2.imread(str(f)) is not None:
                key = f"refl_{len(rows):04d}"
                dst = OUTDIR / f"{key}.jpg"
                dst.write_bytes(f.read_bytes())
                rows.append({"key": key, "image": dst.name, "clip": str(clip.relative_to(REPO)),
                             "video": v, "t": t, "verified": None})
                per_vid[v] += 1
                progressed = True
            subprocess.run(["rm", "-rf", tmp], check=False)
        if not progressed:
            break
    json.dump({"n": len(rows), "n_videos": len({r["video"] for r in rows}),
               "per_clip": per_clip, "seed": seed,
               "leakage_assertion": "thin768 dataset /dataset_seg_thin768: 4965 images, 0 Right_Left",
               "verification_status": "PENDING — no frame may be scored until verified is True",
               "rows": rows}, open(INDEX, "w"), indent=1)
    print(f"staged {len(rows)} frames / {len({r['video'] for r in rows})} videos -> {INDEX}")


def contact_sheet(cols=6, cell=260):
    """Montages for review — each cell labelled with its index so decisions can be recorded."""
    idx = json.load(open(INDEX))
    rows = idx["rows"]
    per_sheet = cols * 5
    made = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        nr = (len(chunk) + cols - 1) // cols
        sheet = np.full((nr * cell, cols * cell, 3), 30, np.uint8)
        for i, r in enumerate(chunk):
            im = cv2.imread(str(OUTDIR / r["image"]))
            if im is None:
                continue
            h, w = im.shape[:2]
            sc = (cell - 24) / max(h, w)
            im = cv2.resize(im, (max(1, int(w * sc)), max(1, int(h * sc))))
            y, x = (i // cols) * cell, (i % cols) * cell
            sheet[y + 20:y + 20 + im.shape[0], x + 4:x + 4 + im.shape[1]] = im
            cv2.putText(sheet, f"{s+i}", (x + 6, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 235, 255), 1, cv2.LINE_AA)
        p = OUTDIR / f"sheet_{s//per_sheet:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made.append(p)
        print(f"  {p}  ({len(chunk)} frames, indices {s}..{s+len(chunk)-1})")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--per-clip", type=int, default=2)
    ap.add_argument("--min-videos", type=int, default=20)
    a = ap.parse_args()
    if a.sample:
        sample(a.n, a.per_clip, a.min_videos)
    if a.contact_sheet:
        contact_sheet()
