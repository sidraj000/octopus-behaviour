"""extract_mask_feats.py — turn the segmenter into features for the ethogram classifier.

WHY THIS AND NOT ANOTHER REGULARISER. R33 established the pattern: everything that adds real
information helps (representation +0.087), everything that reshuffles existing information does not
(head capacity, upsampling, mixup, noise -- all ~0 or negative). Mask geometry is new information:
CLIP/DINOv2/VideoMAE are appearance encoders that never explicitly localise the animal, so nothing in
the current feature stack knows WHERE the octopus is, HOW BIG it is, or whether it MOVED.

It also targets our measured weaknesses rather than generic ones:

  * `No octopus` <-> `Resting` is 22% of all errors. Both are static and their whole-frame motion
    medians are IDENTICAL (0.0198 vs 0.0198), so the existing motion channels provably cannot separate
    them. Mask AREA can: ~0 vs a real blob. The segmenter's presence AUC is 0.794 per-frame and 0.969
    with EMA fusion.
  * `Locomotion` vs `Resting`: CENTROID DISPLACEMENT is the animal's own translation. The current
    `motion_disp` is whole-frame changed pixels, confounded by IR lamp flicker, TVs and people.
  * `Reaching out of water` (a sink class, precision 0.43): centroid HEIGHT in frame is a direct cue.

And the features we take are the accurate part of a mediocre model: SEG-TEST mask IoU is 0.6415, but
AREA ERROR IS ~1%. Boundary quality is the weak axis and we ignore boundaries.

--------------------------------------------------------------------------------------------------
TWO CONSTRAINTS THAT MUST TRAVEL WITH ANY RESULT FROM THIS

1. LEAKAGE, measured not hand-waved. The segmenter trained on 11 of the 34 ethogram TEST videos (and
   56 of 133 train). That is not label leakage -- masks are not ethogram labels -- but its features
   will be sharper on those videos than in deployment, which inflates any gain. Each row therefore
   carries `seg_seen_video`, so the gain can be reported SPLIT by whether the segmenter had seen the
   video, and the unseen-video subset treated as the real number.

2. IR IS 35% OF THE CORPUS AND THE SEGMENTER IS COLOUR-TRAINED. On `Right_Top` it over-segments bright
   metal tools (mask area median 8.5% vs 2.9% on colour) and GroundingDINO accepted only 13% of IR
   clips during labelling -- `batch_skeleton_motion.py` skips IR for exactly this reason. So IR mask
   features are not merely noisy, they are actively misleading. They are ZEROED with `valid=0` rather
   than passed through, because a wrong-but-plausible area is indistinguishable from a real empty tank,
   and with 133 training videos the head should not have to learn to ignore a garbage block.

--------------------------------------------------------------------------------------------------
CHANNELS (per sampled frame, so the block is [T, 10] and appends to any backbone):
   0 area_frac        mask pixels / frame pixels               -- presence + body spread
   1 centroid_x       normalised 0-1                           -- location in tank
   2 centroid_y       normalised 0-1                           -- surface proximity (reaching)
   3 bbox_w           normalised                               -- posture extent
   4 bbox_h           normalised
   5 elongation       max(w,h)/min(w,h)                        -- stretched vs balled
   6 solidity         mask area / bbox area                    -- spread arms vs compact body
   7 masked_motion    changed-pixel fraction INSIDE the mask   -- movement of the animal only
   8 centroid_disp    |centroid - prev sampled centroid|       -- translation => locomotion
   9 valid            1 = colour camera with a usable mask     -- gate for everything above

Output `src/dataset_etho/v1/feats_mask/<clip>.npy` [T, 10] + meta.json. Resumable per clip.

Usage: venv/bin/python3 src/extract_mask_feats.py [--ckpt weights/seg/octo_seg_thin768_lraspp.pt]
"""
import argparse, json, os, queue, shutil, sys, tempfile, threading, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from ensemble_235b import extract_frames_at, DENSE_FPS
from build_ethogram_dataset import ROOTS
from segment_octopus import OctoSegmenter

IR_CAMERA = "Right_Top"
N_CH = 10
MIN_AREA = 1e-5          # below this the mask is treated as empty rather than a tiny blob
MAX_AREA = 0.60          # above this the "mask" is a failure (whole-frame activation), not an animal


def resolve(clip):
    for r in ROOTS:
        p = r / clip
        if p.exists() and p.stat().st_size > 10000:
            return p
    return None


def seg_training_videos():
    """Source videos the segmenter saw, so each row can record whether it is a seen video."""
    vids = set()
    for m in list((REPO / "data").rglob("dataset_seg*/**/manifest.jsonl")) + \
             list((REPO / "data").glob("dataset_seg_human/manifest.jsonl")):
        try:
            for line in open(m):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                c = str(r.get("clip") or r.get("clip_path") or r.get("source") or "")
                p = [x for x in c.split("/") if x]
                if len(p) >= 3:
                    vids.add("/".join(p[-3:-1]))
        except Exception:
            continue
    return vids


def mask_geometry(mask):
    """-> (area_frac, cx, cy, w, h, elongation, solidity) from a binary mask, or None if unusable."""
    h, w = mask.shape
    area = float(mask.sum())
    frac = area / (h * w)
    if frac < MIN_AREA or frac > MAX_AREA:
        return None
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    elong = float(max(bw, bh) / max(1, min(bw, bh)))
    solidity = float(area / (bw * bh))
    return (frac, float(xs.mean() / w), float(ys.mean() / h),
            float(bw / w), float(bh / h), elong, solidity)


def masked_motion(prev_grey, grey, mask, pix_thresh=25):
    """Changed-pixel fraction INSIDE the mask -- movement of the animal, not of the room."""
    if prev_grey is None or prev_grey.shape != grey.shape or not mask.any():
        return 0.0
    d = np.abs(grey.astype(np.float32) - prev_grey.astype(np.float32)) > pix_thresh
    return float((d & mask).sum() / max(1, mask.sum()))


def prefetch(rows, q, stop):
    for r in rows:
        if stop.is_set():
            break
        src = resolve(r["clip"])
        if src is None:
            q.put((r, None, None)); continue
        td = tempfile.mkdtemp(prefix="mf_")
        try:
            fr = extract_frames_at(src, td, DENSE_FPS)
            if fr:
                q.put((r, fr, td))
            else:
                shutil.rmtree(td, ignore_errors=True); q.put((r, None, None))
        except Exception:
            shutil.rmtree(td, ignore_errors=True); q.put((r, None, None))
    q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None, help="force cpu when MPS is busy with another job")
    a = ap.parse_args()

    d = REPO / "src" / "dataset_etho" / a.version
    man = [json.loads(l) for l in open(d / "manifest.jsonl") if l.strip()]
    if a.limit:
        man = man[:a.limit]
    out = d / "feats_mask"; out.mkdir(parents=True, exist_ok=True)
    seg = OctoSegmenter(a.ckpt, device=a.device)
    seen = seg_training_videos()
    print(f"segmenter {Path(a.ckpt).name} | in_size={seg.in_size} arch={seg.arch} device={seg.device}")
    print(f"segmenter training videos known: {len(seen)}")
    n_seen = len({r['video'] for r in man if r['video'] in seen})
    print(f"of this dataset's {len({r['video'] for r in man})} videos, {n_seen} were seen by the segmenter")

    done = {p.stem for p in out.glob("*.npy")}
    todo = [r for r in man if r["clip"].replace("/", "__") not in done]
    print(f"clips: {len(man)}  todo: {len(todo)}  (IR clips get a zeroed block with valid=0)")
    if not todo:
        print("nothing to do"); return

    q, stop = queue.Queue(maxsize=a.workers * 2), threading.Event()
    chunks = [todo[i::a.workers] for i in range(a.workers)]
    threads = [threading.Thread(target=prefetch, args=(c, q, stop), daemon=True) for c in chunks]
    for t in threads:
        t.start()

    t0, seen_n, live, ok, fail, ir = time.time(), 0, len(threads), 0, 0, 0
    stats = []
    try:
        while live:
            item = q.get()
            if item is None:
                live -= 1; continue
            r, fr, td = item
            seen_n += 1
            try:
                pick = [i for i in (r.get("frames_used") or []) if fr and i < len(fr)]
                T = len(r.get("frames_used") or []) or 10
                arr = np.zeros((T, N_CH), np.float32)
                if r.get("camera") == IR_CAMERA:
                    ir += 1                                   # zeroed block, valid stays 0
                elif fr and pick:
                    # ONE batched forward for the clip: every sampled frame is the same size, and
                    # segment_batch is documented to give per-frame-identical results.
                    imgs = [np.asarray(Image.open(fr[i]).convert("RGB")) for i in pick]
                    masks = seg.segment_batch([im[:, :, ::-1] for im in imgs])
                    greys = []
                    for im in imgs:
                        g_ = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                        gh, gw = g_.shape
                        g_[int(gh * 0.88):, int(gw * 0.60):] = 0   # same timestamp mask as elsewhere
                        greys.append(g_)
                    prev_c, prev_grey = None, None
                    for j, ((mask, _area), grey) in enumerate(zip(masks, greys)):
                        mask = np.asarray(mask, bool)
                        g = mask_geometry(mask)
                        if g is None:
                            prev_grey = grey
                            continue                          # leave this timestep zero/invalid
                        frac, cx, cy, bw, bh, elong, sol = g
                        mm = masked_motion(prev_grey, grey, mask)
                        disp = 0.0 if prev_c is None else float(np.hypot(cx - prev_c[0], cy - prev_c[1]))
                        arr[j] = [frac, cx, cy, bw, bh, elong, sol, mm, disp, 1.0]
                        prev_c, prev_grey = (cx, cy), grey
                    if arr[:, 9].any():
                        stats.append(float(arr[arr[:, 9] > 0, 0].mean()))
                np.save(out / (r["clip"].replace("/", "__") + ".npy"), arr)
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 3:
                    print(f"  FAIL {r['clip']}: {type(e).__name__}: {e}")
            finally:
                if td:
                    shutil.rmtree(td, ignore_errors=True)
            if seen_n % 200 == 0:
                rate = seen_n / max(time.time() - t0, 1e-9) * 60
                print(f"  {seen_n}/{len(todo)}  {rate:.0f} clips/min  eta {(len(todo)-seen_n)/max(rate,1e-9):.0f} min"
                      f"  ok={ok} ir_zeroed={ir} fail={fail}"
                      + (f"  median area {np.median(stats):.4f}" if stats else ""), flush=True)
    except KeyboardInterrupt:
        stop.set(); print("\ninterrupted -- resumable")

    (out / "meta.json").write_text(json.dumps({
        "ckpt": Path(a.ckpt).name, "in_size": seg.in_size, "arch": seg.arch,
        "feat_dim": N_CH, "n_motion": 0, "n_clips": len(list(out.glob("*.npy"))),
        "n_ir_zeroed": ir, "n_failed": fail,
        "channels": ["area_frac", "centroid_x", "centroid_y", "bbox_w", "bbox_h", "elongation",
                     "solidity", "masked_motion", "centroid_disp", "valid"],
        "median_area_frac_valid": round(float(np.median(stats)), 5) if stats else None,
        "seg_seen_videos_in_dataset": n_seen,
        "caveats": ["segmenter trained on 11/34 ethogram TEST videos -- report gains split by "
                    "seg_seen_video and treat the unseen subset as the real number",
                    "IR (Right_Top) is zeroed with valid=0: the colour-trained segmenter "
                    "over-segments bright tools on IR, so its output there is misleading not noisy"],
    }, indent=1))
    print(f"\nwrote {out}  clips={len(list(out.glob('*.npy')))}  ir_zeroed={ir}  failed={fail}")
    if stats:
        print(f"median mask area on valid frames: {np.median(stats):.4f} "
              f"(colour-clip expectation ~0.029 from SEGMENTATION_LOG)")


if __name__ == "__main__":
    main()
