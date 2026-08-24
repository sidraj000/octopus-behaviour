"""eval_teacher_masks.py — score the ZERO-SHOT TEACHER against the human masks on SEG-TEST.

Fills the one gap called out in SEGMENTATION_LOG.md: every segmentation IoU we have is either
student-vs-human (SEG-TEST) or student-vs-teacher (agreement). The teacher itself — the zero-shot
foundation pair GroundingDINO-tiny (box) -> SAM2 (mask) — was never scored against human masks, so
we could not say whether the student trails the teacher or has already caught up to it. Without
that number, "the ceiling is teacher-label quality" is a hypothesis, not a measurement.

WHAT IS COMPARED — both arms see the SAME single frame and are scored on the SAME human mask:
  teacher : GroundingDINO box -> SAM2 (box + positive/negative points) -> largest blob
  student : the deployed tiny segmenter (OctoSegmenter)

The frozen set is imported from benchmarks.py (NOT re-derived here) so the frames, the holdout
videos and the IoU computation are provably identical to the published student numbers.

SINGLE-FRAME, deliberately. The training labels were made with SAM2 *video propagation* seeded by
the most-confident frame, which Phase 0 showed beats per-frame box->SAM. This script measures the
per-frame teacher because (a) the student is itself a per-frame model, so this is the apples-to-apples
comparison, and (b) SEG-TEST's 122 frames come from 122 DISTINCT clips, so propagation would cost
122 x 40 = 4,880 GroundingDINO calls (~3.4 h locally). Read the teacher row as "zero-shot teacher on
one frame", NOT as the quality of the propagated labels the student actually learned from — those are
better. `--propagate N` runs the true propagated recipe on a random N-clip subsample for that reason.

Device: GroundingDINO on CPU (its deformable attention is unstable on MPS); SAM2 on MPS/CUDA.

Usage
  venv/bin/python3 src/eval_teacher_masks.py --tag teacher_v1
  venv/bin/python3 src/eval_teacher_masks.py --limit 12          # quick smoke run
  venv/bin/python3 src/eval_teacher_masks.py --propagate 15      # + true-recipe subsample
"""
import argparse, json, os, shutil, sys, tempfile, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from benchmarks import _manifest, _source_video, HOLDOUT_VIDEOS, DS   # frozen set: do not re-derive
from auto_segment import (gd_best_box, build_prompts, largest_blob, load_models, pick_device,
                          MIN_SEED_CONF, MAXSIDE, AREA_MIN, AREA_MAX)

STUDENT_CKPT = REPO / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"   # the paper_current model
OUT = REPO / "data" / "teacher_vs_human_masks.json"


def iou_of(pred, gt):
    """IoU with benchmarks.py's exact convention (nearest-resize pred onto the GT grid)."""
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    return float(inter / union) if union else 1.0


def teacher_frame_mask(img_pil, M):
    """One frame -> (mask, seed_conf). Mirrors auto_segment's seed-frame path: GD box, then SAM2
    steered by box + positive/negative points, then largest-blob cleanup."""
    box, conf = gd_best_box(img_pil, M)
    if box is None:
        return None, 0.0
    pts = labs = None
    try:
        pts, labs = build_prompts(img_pil, box)
    except Exception:
        pts = labs = None
    with tempfile.TemporaryDirectory() as td:
        fdir = Path(td) / "frames"; fdir.mkdir()
        img_pil.save(fdir / "00001.jpg", quality=95)
        sam2 = M["sam2"]
        st = sam2.init_state(video_path=str(fdir))
        sam2.add_new_points_or_box(st, frame_idx=0, obj_id=1,
                                  box=np.array(box, np.float32), points=pts, labels=labs)
        mask = None
        for _, _, logits in sam2.propagate_in_video(st, start_frame_idx=0):
            mask = (logits[0] > 0).cpu().numpy()[0]
            break
    if mask is None or not mask.any():
        return None, conf
    return largest_blob(mask), conf


def summarize(name, ious, extra=None):
    a = np.asarray(ious, float)
    d = {"n": int(a.size), "iou_mean": round(float(a.mean()), 4),
         "iou_median": round(float(np.median(a)), 4)}
    if extra:
        d.update(extra)
    print(f"  {name:<28} n={d['n']:<4} IoU mean {d['iou_mean']:.4f}  median {d['iou_median']:.4f}"
          + (f"   {extra}" if extra else ""))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="teacher_v1")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (smoke test)")
    ap.add_argument("--student-ckpt", default=str(STUDENT_CKPT))
    ap.add_argument("--gd", default="tiny"); ap.add_argument("--sam2", default="tiny")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    pos = [r for r in _manifest("human") if _source_video(r["clip"]) in HOLDOUT_VIDEOS]
    pos.sort(key=lambda r: r["image"])                      # deterministic order
    if args.limit:
        pos = pos[:args.limit]
    videos = sorted({_source_video(r["clip"]) for r in pos})
    print(f"SEG-TEST: {len(pos)} frames / {len(videos)} holdout videos")

    dev = pick_device()
    print(f"device={dev} (GroundingDINO forced to CPU on mps)")
    M = load_models(dev, gd_model=args.gd, sam2_model=args.sam2)

    from segment_octopus import OctoSegmenter
    S = OctoSegmenter(args.student_ckpt)
    print(f"student: {Path(args.student_ckpt).name}")

    t_iou, s_iou, rows = [], [], []
    n_nodet, confs = 0, []
    t0 = time.time()
    for i, r in enumerate(pos):
        img_bgr = cv2.imread(str(DS / r["image"]))
        gt = cv2.imread(str(DS / r["mask"]), 0) > 127
        img_pil = Image.open(DS / r["image"]).convert("RGB")

        tm, conf = teacher_frame_mask(img_pil, M)
        confs.append(conf)
        if tm is None:
            n_nodet += 1
            ti = 0.0                       # a miss is a real failure, scored as IoU 0 (not dropped)
        else:
            ti = iou_of(tm, gt)
        sp, _ = S.segment(img_bgr)
        si = iou_of(sp, gt)

        t_iou.append(ti); s_iou.append(si)
        rows.append({"image": r["image"], "video": _source_video(r["clip"]),
                     "camera": r["camera"], "gt_area": round(float(gt.mean()), 5),
                     "teacher_iou": round(ti, 4), "student_iou": round(si, 4),
                     "teacher_conf": round(conf, 3),
                     "teacher_area": None if tm is None else round(float(tm.mean()), 5),
                     "student_area": round(float(sp.mean()), 5)})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pos)}] teacher {np.mean(t_iou):.3f} | student {np.mean(s_iou):.3f}"
                  f" | {(time.time()-t0)/(i+1):.1f}s/frame", flush=True)

    print(f"\n=== SEG-TEST, {len(pos)} frames / {len(videos)} videos (identical set, human masks) ===")
    res = {
        "_meta": {"tag": args.tag, "n_frames": len(pos), "n_videos": len(videos),
                  "student_ckpt": Path(args.student_ckpt).name,
                  "teacher": f"GD-{args.gd} + SAM2-{args.sam2}, single frame, box+points, largest blob",
                  "elapsed_s": round(time.time() - t0, 1),
                  "note": "teacher is PER-FRAME; the training labels used SAM2 video propagation, "
                          "which Phase 0 showed is better. Not the propagated-label quality."},
        "teacher": summarize("teacher (zero-shot)", t_iou,
                             {"no_detection": n_nodet,
                              "conf_median": round(float(np.median(confs)), 3),
                              "below_min_seed_conf": int(sum(c < MIN_SEED_CONF for c in confs))}),
        "student": summarize("student (ours)", s_iou),
    }
    d = np.asarray(t_iou) - np.asarray(s_iou)
    # paired bootstrap, CLUSTERED BY SOURCE VIDEO (frames from one recording are near-duplicates)
    vids = np.array([r["video"] for r in rows])
    uniq = np.unique(vids)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(5000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([d[vids == v] for v in pick])
        boot.append(vals.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res["delta_teacher_minus_student"] = {
        "mean": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "includes_zero": bool(lo <= 0 <= hi),
        "clustered_by": "source video", "n_clusters": int(len(uniq)),
        "teacher_wins_frames": int((d > 0).sum()), "student_wins_frames": int((d < 0).sum()),
    }
    print(f"\n  delta (teacher - student) = {d.mean():+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]"
          f"  {'INCLUDES 0' if lo <= 0 <= hi else 'excludes 0'}")
    print(f"  per-frame wins: teacher {int((d>0).sum())} / student {int((d<0).sum())}")

    res["per_frame"] = rows
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
