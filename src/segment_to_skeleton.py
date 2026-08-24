"""segment_to_skeleton.py — bridge our octopus segmentation -> the anatomical skeleton pipeline.

Takes a local clip (or video), runs the deployed mask model (segment_octopus) with temporal EMA
smoothing over frames, keeps the octopus-present frames, crops them all to a SINGLE fixed union
bbox (so the octopus fills the frame for a clean skeleton AND node coordinates stay consistent
across frames -> valid motion metadata), writes binary silhouettes, then runs the multi-frame
skeletonizer (src/skeleton/) to produce per-frame anatomical graphs + motion_metadata.csv.

Usage:
  venv/bin/python3 src/segment_to_skeleton.py <clip.mp4> <out_dir> [--fps 5] [--ckpt weights/seg/...]
                    [--present 0.004] [--single]   # --single: just skeleton the best frame
"""
import argparse, sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter, _largest_blob

DEFAULT_CKPT = HERE.parent / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"
EMA_ALPHA = 0.45
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def segment_masks(clip, S, fps, present_frac, keep_small=0, refine=""):
    """Return (kept_masks[HxW bool at native res], src_fps, sample_step). EMA-smoothed, largest-blob.

    keep_small > 0: ALSO return a 4th element — downscaled colour frames (width=keep_small) aligned
    with the masks, used to build grey crops for the optical-flow tracking prior (native 4K greys
    for a whole clip would be ~1GB; small colour is ~1MB/frame)."""
    cap = cv2.VideoCapture(str(clip))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps)))
    masks, smalls, i, ema = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            prob = S.prob(frame)                                   # low-res prob
            ema = prob if ema is None else EMA_ALPHA * prob + (1 - EMA_ALPHA) * ema
            H, W = frame.shape[:2]
            m = cv2.resize(ema, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
            if m.any():
                m = _largest_blob(m)
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)
            if refine == "sam2" and m.any() and m.mean() >= present_frac:
                from mask_refine import sam2_refine   # offline: +0.44 arms, better tips (bench50)
                m = sam2_refine(frame, m, largest_blob=_largest_blob)
            masks.append(m if m.mean() >= present_frac else None)
            if keep_small:
                smalls.append(cv2.resize(frame, (keep_small, int(round(H * keep_small / W))),
                                         interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    if keep_small:
        return masks, src_fps, step, smalls
    return masks, src_fps, step


def grey_crops(smalls, masks_shape, bbox, indices):
    """Grey crop (resized to the mask-crop's shape) for each index, from the small colour frames."""
    H, W = masks_shape
    y0, y1, x0, x1 = bbox
    out = []
    for k in indices:
        sm = smalls[k]
        s = sm.shape[1] / float(W)
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        gc = g[int(y0 * s):max(int(y1 * s), int(y0 * s) + 1), int(x0 * s):max(int(x1 * s), int(x0 * s) + 1)]
        out.append(cv2.resize(gc, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR))
    return out


def union_bbox(masks, pad_frac=0.12):
    ys, xs = [], []
    for m in masks:
        if m is None or not m.any():
            continue
        yy, xx = np.where(m)
        ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    if not ys:
        return None
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    H = y1 - y0; W = x1 - x0; py = int(H * pad_frac) + 4; px = int(W * pad_frac) + 4
    return max(0, y0 - py), y1 + py, max(0, x0 - px), x1 + px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--present", type=float, default=0.004, help="min mask area frac to count as present")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--min-arms", type=int, default=3, help="min distinct arms (real curled/occluded "
                    "poses in 2D often show fewer than 8 tips; 3 is realistic for our masks)")
    ap.add_argument("--max-arms", type=int, default=8)
    ap.add_argument("--single", action="store_true", help="skeleton only the single best (largest-mask) frame")
    ap.add_argument("--refine", default="", choices=["", "sam2"],
                    help="offline mask refinement: 'sam2' = student locates, SAM2 sharpens "
                         "(+0.44 arms & better tips on bench50; ~1-2s/frame)")
    args = ap.parse_args()

    out = Path(args.out); (out / "masks").mkdir(parents=True, exist_ok=True)
    print(f"segmenting {args.clip} with {Path(args.ckpt).name} …", flush=True)
    S = OctoSegmenter(args.ckpt)
    masks, src_fps, step = segment_masks(args.clip, S, args.fps, args.present, refine=args.refine)
    present = [(k, m) for k, m in enumerate(masks) if m is not None]
    print(f"  {len(masks)} sampled frames, {len(present)} octopus-present", flush=True)
    if not present:
        print("no octopus-present frames — nothing to skeletonize"); return

    bb = union_bbox([m for _, m in present])
    y0, y1, x0, x1 = bb
    print(f"  fixed crop bbox: rows {y0}:{y1}  cols {x0}:{x1}  ({y1-y0}x{x1-x0})", flush=True)

    kept = []
    for k, m in present:
        crop = (m[y0:y1, x0:x1].astype(np.uint8)) * 255
        p = out / "masks" / f"{k:05d}.png"
        cv2.imwrite(str(p), crop); kept.append(p)

    # effective fps of the kept sample stream (for motion dt)
    eff_fps = src_fps / step

    if args.single:
        best = max(present, key=lambda km: km[1].mean())[0]
        import skeleton
        print(f"  single-frame skeleton on frame {best} …", flush=True)
        skeleton.run(str(out / "masks" / f"{best:05d}.png"), str(out / "skeleton_single"),
                     iterations=3, min_arms=args.min_arms, max_arms=args.max_arms)
    else:
        import multi_frame
        print(f"  multi-frame skeleton over {len(kept)} masks (fps={eff_fps:.2f}) …", flush=True)
        multi_frame.run_sequence(str(out / "masks"), str(out / "skeleton_seq"),
                                 stride=1, fps=eff_fps, video_fps=max(2.0, args.fps),
                                 iterations=2, min_arms=args.min_arms, max_arms=args.max_arms,
                                 edit_first=False)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
