"""skel_diagnostic.py — Phase 1: is skeleton quality limited by the MASK or the skeletonizer?

For a sample of human-GT frames, run the single-frame skeletonizer on BOTH the clean human GT mask
and our seg-model's mask for the same image, and compare the resulting arm counts. If GT masks give
~8 arms and model masks give ~4, the bottleneck is mask quality (tentacle recall), not the
skeletonizer. Writes side-by-side overlay images + summary.json to data/skel_diag/.
"""
import sys, json
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter, _largest_blob
from multi_frame import process_frame
from seg_skeleton_pipeline import _draw_skeleton, DEFAULT_CKPT

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
MAXDIM = 1024


def skel(mask_bool):
    """Arm count + graph for a boolean mask (min_arms=1 so it never fatally raises)."""
    m = (mask_bool.astype(np.uint8)) * 255
    try:
        nodes, edges, metrics, _ = process_frame(m, 3, MAXDIM, 1, 8, None)
        return int(metrics["arm_count"]), nodes, edges
    except Exception:
        return 0, None, None


def _panel(base, nodes, edges, title, arms):
    c = base.copy()
    if nodes:
        _draw_skeleton(c, nodes, edges, 2)
    cv2.rectangle(c, (0, 0), (c.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(c, f"{title}: {arms} arms", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return c


def main(n=40):
    OUT.mkdir(parents=True, exist_ok=True)
    S = OctoSegmenter(str(DEFAULT_CKPT))
    rows = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("source") == "human" and r.get("image")]
    idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    summ = []
    for j, i in enumerate(idx):
        r = rows[int(i)]
        img = cv2.imread(str(DS / r["image"])); gt = cv2.imread(str(DS / r["mask"]), 0) > 127
        mm, _ = S.segment(img)
        a_gt, n_gt, e_gt = skel(gt)
        a_mm, n_mm, e_mm = skel(mm)
        dim = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
        left = _panel(dim, n_gt, e_gt, "GT mask", a_gt)
        right = _panel(dim, n_mm, e_mm, "MODEL mask", a_mm)
        gap = np.full((left.shape[0], 6, 3), 40, np.uint8)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), np.hstack([left, gap, right]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        summ.append({"file": f"{j:03d}.jpg", "gt_arms": a_gt, "model_arms": a_mm, "clip": r.get("clip")})
        print(f"  [{j+1}/{len(idx)}] GT {a_gt} vs MODEL {a_mm} arms", flush=True)
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    ga = np.array([s["gt_arms"] for s in summ]); ma = np.array([s["model_arms"] for s in summ])
    print(f"\nGT mask arms   : mean {ga.mean():.2f}  median {int(np.median(ga))}  >=6: {int((ga>=6).sum())}/{len(ga)}")
    print(f"MODEL mask arms: mean {ma.mean():.2f}  median {int(np.median(ma))}  >=6: {int((ma>=6).sum())}/{len(ma)}")
    print(f"delta (GT-MODEL) mean arms: {ga.mean()-ma.mean():+.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
