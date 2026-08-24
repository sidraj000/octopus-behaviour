#!/usr/bin/env python3
"""eval_reflection_presence.py — does the DEPLOYED segmenter reject reflections?

The paper reports presence AUC 0.794 for thin768 and describes the system as reflection-robust, but
that AUC came from 19 empty-TANK negatives on the same cameras as the positives. The reflection
failure mode (Right_Left: the camera sees the room and a mirrored human through the glass, and the
CLIP detector fires at p_visible=1.0) was measured for the v3 negatives model, never for thin768.

This scores thin768's mask area as a presence signal against BOTH negative types, reported
SEPARATELY — pooling them would silently redefine the metric:

  empty-tank negatives  same cameras as the positives, human-labelled  (SEG-TEST's 19)
  reflection negatives  Right_Left, leak-free by construction          (this study)

Leakage assertion (verified): thin768's dataset /dataset_seg_thin768 contains 4,965 images and
**0** Right_Left frames.

Statistics: frames from one recording are near-duplicates, so n is the number of VIDEOS, not frames.
Confidence intervals are cluster-bootstrapped BY SOURCE VIDEO (same discipline as kinematics_stats.py).

Headline metric is FP rate at a fixed present-recall (the operating point the extraction gate uses),
with AUC secondary.
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from segment_octopus import OctoSegmenter, _largest_blob

DS = REPO / "data" / "dataset_seg_human"
CACHE = REPO / "data" / "fusion_probcache"
REFL = REPO / "data" / "reflection_negatives"
OUT = REPO / "data" / "reflection_presence.json"


def areas_from_cache(mode, thresh):
    """Positive + empty-tank negative mask areas from the prob cache (already computed).

    BUG FIXED 2026-08-15: `video` used to be the image FILENAME, so the cluster bootstrap treated
    all 122 positives as 122 independent recordings when they come from just 5. That understates
    clustering and yields CIs that are too NARROW (the reflection CI was reported as [0.871, 0.964];
    with the true 5-video grouping it is [0.826, 0.966]). Point estimates are unaffected — only the
    uncertainty was wrong, and in the flattering direction.
    """
    import benchmarks as B
    d = CACHE / mode
    idx = json.load(open(d / "index.json"))
    pos_src = [r for r in B._manifest("human") if B._source_video(r["clip"]) in B.HOLDOUT_VIDEOS]
    neg_src = [r for r in B._manifest("negative") if B._source_video(r["clip"]) in B.HOLDOUT_VIDEOS]
    pos, neg = [], []
    for r in idx["rows"]:
        pr = np.load(d / f"{r['key']}.npz")["prob"].astype(np.float32)
        H, W = r["shape"]
        m = cv2.resize(pr, (W, H), interpolation=cv2.INTER_LINEAR) > thresh
        if m.any():
            m = _largest_blob(m)
        bucket, src = (pos, pos_src) if r["label"] == "pos" else (neg, neg_src)
        i = len(bucket)
        vid = B._source_video(src[i]["clip"]) if i < len(src) else "unknown"
        bucket.append({"area": float(m.mean()), "video": vid})
    return pos, neg


def reflection_areas(S, thresh):
    idx = json.load(open(REFL / "index.json"))
    rows = [r for r in idx["rows"] if r.get("verified") is True]
    out = []
    for r in rows:
        img = cv2.imread(str(REFL / r["image"]))
        pr = S.prob(img)
        m = cv2.resize(pr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) > thresh
        if m.any():
            m = _largest_blob(m)
        out.append({"area": float(m.mean()), "video": r["video"]})
    return out


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    sc = np.r_[[p["area"] for p in pos], [n["area"] for n in neg]]
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1, n0 = lab.sum(), len(lab) - lab.sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fp_at_recall(pos, neg, recall):
    cut = float(np.quantile([p["area"] for p in pos], 1 - recall))
    return float(np.mean([n["area"] >= cut for n in neg])), cut


def fp_at_area(neg, area=0.01):
    """The deployed operating point: the extraction gate fires when mask area >= 0.01."""
    return float(np.mean([n["area"] >= area for n in neg]))


def boot_auc_by_video(pos, neg, iters=2000, seed=5):
    """Cluster bootstrap: resample VIDEOS with replacement, not frames."""
    rng = np.random.default_rng(seed)
    pv, nv = {}, {}
    for p in pos:
        pv.setdefault(p["video"], []).append(p)
    for n in neg:
        nv.setdefault(n["video"], []).append(n)
    pk, nk = list(pv), list(nv)
    vals = []
    for _ in range(iters):
        P = [x for k in rng.choice(pk, len(pk)) for x in pv[k]]
        N = [x for k in rng.choice(nk, len(nk)) for x in nv[k]]
        vals.append(auc(P, N))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "weights/seg/octo_seg_thin768_lraspp.pt"))
    ap.add_argument("--mode", default="none", help="fusion mode whose prob cache to use for positives")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()

    S = OctoSegmenter(a.ckpt)
    pos, tank = areas_from_cache(a.mode, a.thresh)
    refl = reflection_areas(S, a.thresh)
    print(f"positives {len(pos)}   empty-tank negs {len(tank)}   "
          f"reflection negs {len(refl)} / {len({r['video'] for r in refl})} videos\n")

    res = {"mode": a.mode, "thresh": a.thresh, "ckpt": Path(a.ckpt).name,
           "n_pos": len(pos), "n_tank": len(tank), "n_refl": len(refl),
           "n_refl_videos": len({r["video"] for r in refl})}
    for name, neg in (("empty_tank", tank), ("reflection", refl)):
        A = auc(pos, neg)
        f90, cut90 = fp_at_recall(pos, neg, 0.90)
        f80, _ = fp_at_recall(pos, neg, 0.80)
        lo, hi = boot_auc_by_video(pos, neg)
        res[name] = {"auc": round(A, 4), "auc_ci95_by_video": [round(lo, 4), round(hi, 4)],
                     "fp_at_recall90": round(f90, 4), "fp_at_recall80": round(f80, 4),
                     "area_cut_at_recall90": round(cut90, 5),
                     "fp_at_deployed_area_0.01": round(fp_at_area(neg), 4),
                     "median_neg_area": round(float(np.median([n["area"] for n in neg])), 5)}
        r = res[name]
        print(f"{name:12s} AUC {r['auc']:.4f}  CI95[{lo:.3f},{hi:.3f}]  "
              f"FP@R90 {r['fp_at_recall90']:.3f}  FP@R80 {r['fp_at_recall80']:.3f}  "
              f"FP@area>=.01 {r['fp_at_deployed_area_0.01']:.3f}  "
              f"med.neg.area {r['median_neg_area']:.4f}")
    print(f"\npositive median area {np.median([p['area'] for p in pos]):.4f}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
