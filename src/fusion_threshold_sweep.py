#!/usr/bin/env python3
"""fusion_threshold_sweep.py — is R8's fusion result a THRESHOLD artifact?

R8 measured single-frame vs flow-fused segmentation at the shipped threshold 0.5 and concluded
(a) fusion HURTS mask IoU (0.642 -> 0.511) and (b) fusion HELPS presence AUC (0.794 -> 0.950).

Both conclusions are suspect for the same reason: a per-pixel MEDIAN over 5 warped probability
maps is not calibrated like a single map. Median-of-neighbours suppresses any pixel that is not
confidently octopus in the majority of frames, which systematically SHRINKS the probability mass.
Comparing the two at a fixed 0.5 threshold therefore compares fusion at a handicapped operating
point, and a negative published on that basis would be wrong.

This script removes the confound: it caches each frame's probability map ONCE per fusion mode,
then sweeps the binarisation threshold, so each arm is scored at ITS OWN best threshold.

  - If fusion at its best threshold still loses on IoU  -> R8's negative is REAL (publishable).
  - If fusion recovers to ~baseline at a lower threshold -> R8's negative is an ARTIFACT and must
    be retracted from PAPER_NOTES before it reaches the paper.

The same sweep is applied to presence separation, reported both as rank-AUC and as the metric the
pipeline actually operates on: false-positive rate at a fixed present-recall.

Usage:
  venv/bin/python3 src/fusion_threshold_sweep.py --modes none,flow
  venv/bin/python3 src/fusion_threshold_sweep.py --modes none,flow --cache-only
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import benchmarks as B                                    # frozen holdout + manifest loader
from segment_octopus import OctoSegmenter, _largest_blob
from temporal_fusion import fused_prob

DS = REPO / "data" / "dataset_seg_human"
CACHE = REPO / "data" / "fusion_probcache"
OUT = REPO / "data" / "fusion_threshold_sweep.json"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def build_cache(mode, ckpt):
    """Compute and store one probability map per benchmark frame for this fusion mode."""
    d = CACHE / mode
    d.mkdir(parents=True, exist_ok=True)
    S = OctoSegmenter(str(ckpt))
    pos = [r for r in B._manifest("human") if B._source_video(r["clip"]) in B.HOLDOUT_VIDEOS]
    neg = [r for r in B._manifest("negative") if B._source_video(r["clip"]) in B.HOLDOUT_VIDEOS]
    rows, n_fail = [], 0
    for label, recs in (("pos", pos), ("neg", neg)):
        for i, r in enumerate(recs):
            key = f"{label}_{i:04d}"
            f = d / f"{key}.npz"
            img = cv2.imread(str(DS / r["image"]))
            if not f.exists():
                if mode == "none":
                    pr, ok = S.prob(img), True
                else:
                    pr, info = fused_prob(S, r["clip"], r.get("seed_frame"), img, mode=mode)
                    ok = info["ok"]
                    n_fail += (not ok)
                np.savez_compressed(f, prob=pr.astype(np.float16), ok=np.array([ok]))
            rows.append({"key": key, "label": label, "image": r["image"],
                         "mask": r.get("mask"), "shape": list(img.shape[:2])})
            if (len(rows)) % 20 == 0:
                print(f"  [{mode}] cached {len(rows)}/{len(pos)+len(neg)}", flush=True)
    json.dump({"mode": mode, "rows": rows, "align_fail": n_fail},
              open(d / "index.json", "w"), indent=1)
    print(f"[{mode}] cache complete: {len(rows)} frames, {n_fail} align failures", flush=True)


def sweep(mode):
    d = CACHE / mode
    idx = json.load(open(d / "index.json"))
    res = []
    for t in THRESHOLDS:
        ious, aerr, pos_area, neg_area = [], [], [], []
        for r in idx["rows"]:
            pr = np.load(d / f"{r['key']}.npz")["prob"].astype(np.float32)
            H, W = r["shape"]
            m = cv2.resize(pr, (W, H), interpolation=cv2.INTER_LINEAR) > t
            if m.any():
                m = _largest_blob(m)
            if r["label"] == "pos":
                gt = cv2.imread(str(DS / r["mask"]), 0) > 127
                if m.shape != gt.shape:
                    m = cv2.resize(m.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0
                u = (m | gt).sum()
                ious.append((m & gt).sum() / u if u else 1.0)
                aerr.append(abs(m.mean() - gt.mean()))
                pos_area.append(float(m.mean()))
            else:
                neg_area.append(float(m.mean()))
        res.append({"thresh": t,
                    "iou_mean": round(float(np.mean(ious)), 4),
                    "iou_median": round(float(np.median(ious)), 4),
                    "area_err_pct": round(float(np.mean(aerr)) * 100, 3),
                    "presence_auc": round(_auc(pos_area, neg_area), 4),
                    "fp_at_r90": round(_fp_at_recall(pos_area, neg_area, 0.90), 4),
                    "fp_at_r80": round(_fp_at_recall(pos_area, neg_area, 0.80), 4)})
        print(f"  [{mode}] t={t:.2f}  IoU {res[-1]['iou_mean']:.4f}  "
              f"AUC {res[-1]['presence_auc']:.4f}  FP@R90 {res[-1]['fp_at_r90']:.3f}", flush=True)
    return res


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1, n0 = lab.sum(), len(lab) - lab.sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fp_at_recall(pos, neg, recall):
    """Area cutoff giving >= `recall` of positives; report the fraction of negatives above it.
    This is the operating point the extraction gate actually uses."""
    if not pos or not neg:
        return float("nan")
    cut = float(np.quantile(pos, 1 - recall))
    return float(np.mean(np.array(neg) >= cut))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="none,flow")
    ap.add_argument("--ckpt", default=str(REPO / "weights/seg/octo_seg_thin768_lraspp.pt"))
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    for m in modes:
        build_cache(m, a.ckpt)
    if a.cache_only:
        sys.exit(0)
    out = {m: sweep(m) for m in modes}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")
    for m in modes:
        best = max(out[m], key=lambda r: r["iou_mean"])
        bauc = max(out[m], key=lambda r: r["presence_auc"])
        print(f"{m:6s} best IoU {best['iou_mean']:.4f} @ t={best['thresh']}   "
              f"best AUC {bauc['presence_auc']:.4f} @ t={bauc['thresh']}")
