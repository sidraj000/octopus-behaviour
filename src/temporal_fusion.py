"""temporal_fusion.py — test-time temporal fusion of the segmenter's probability map.

Tests R3c's untested assertion that the mask model's failure ("right-SIZED but mis-LOCALIZED")
needs a *temporal* fix, and closes a reporting defect: every published IoU is single-frame, but the
deployed skeleton path (segment_to_skeleton.py) thresholds an EMA-smoothed probability map.

Three modes:
  none   single frame (what the paper currently reports)
  ema    causal EMA from clip start up to the target frame — a PROXY for the deployed config
         (deployment runs EMA at fps=3/5 on its own decode; here it is EMA @ 2 fps, see below)
  flow   neighbours t±1, t±2 warped onto t with DIS optical flow, fused by per-pixel median
  median CONTROL for `flow`: the identical neighbours and the identical per-pixel median, but with
         NO optical-flow warping. Any effect present in both is attributable to multi-frame
         averaging alone; only the flow-minus-median difference is attributable to motion
         compensation. Without this control, "optical flow helps" is an unearned mechanism claim.

CRITICAL alignment note (this is why frames are regenerated rather than decoded directly):
`seed_frame` in the label manifest indexes the frame list produced by ui/seg_label.py with
    ffmpeg -vf "fps=2,scale='min(1024,iw)':-2"
NOT raw video frames (clips are 4K at ~12 fps). Decoding raw frame `seed_frame±k` would sample
completely different times. We therefore re-run the identical extraction and assert that the
regenerated frame at `seed_frame` matches the stored labelled image before using its neighbours.
"""
import glob, os, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np

FPS = 2                     # must match ui/seg_label.py
MAXSIDE = 1024              # must match ui/seg_label.py
EMA_ALPHA = 0.45            # matches segment_to_skeleton.py
ALIGN_TOL = 12.0            # mean abs pixel diff allowed between regenerated and stored frame


def extract_same_as_labeler(clip, tmp):
    """Reproduce the labelling tool's frame list exactly."""
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                    f"fps={FPS},scale='min({MAXSIDE},iw)':-2", f"{tmp}/%05d.jpg"], check=False)
    return sorted(glob.glob(f"{tmp}/*.jpg"))


def _align_err(regen_bgr, stored_bgr):
    if regen_bgr is None or stored_bgr is None:
        return 1e9
    if regen_bgr.shape != stored_bgr.shape:
        regen_bgr = cv2.resize(regen_bgr, (stored_bgr.shape[1], stored_bgr.shape[0]))
    return float(np.mean(np.abs(regen_bgr.astype(np.int16) - stored_bgr.astype(np.int16))))


def fused_prob(S, clip, seed_frame, stored_img, mode="none", span=2):
    """Return (prob_map_at_stored_img_size, info). info['ok'] is False if alignment failed."""
    if mode == "none":
        return S.prob(stored_img), {"ok": True, "mode": "none", "n_used": 1}
    tmp = tempfile.mkdtemp()
    try:
        files = extract_same_as_labeler(clip, tmp)
        if not files or seed_frame is None or seed_frame >= len(files):
            return S.prob(stored_img), {"ok": False, "reason": "frame list too short",
                                        "mode": mode, "n_used": 1}
        err = _align_err(cv2.imread(files[seed_frame]), stored_img)
        if err > ALIGN_TOL:                      # regenerated frame is not the labelled frame
            return S.prob(stored_img), {"ok": False, "reason": f"align_err={err:.1f}",
                                        "mode": mode, "n_used": 1}
        H, W = stored_img.shape[:2]

        if mode == "ema":                        # causal: clip start -> seed frame
            ema = None
            for i in range(0, seed_frame + 1):
                p = S.prob(cv2.imread(files[i]))
                ema = p if ema is None else EMA_ALPHA * p + (1 - EMA_ALPHA) * ema
            return ema, {"ok": True, "mode": "ema", "n_used": seed_frame + 1, "align_err": err}

        # mode in ("flow", "median"): fuse neighbours by per-pixel median.
        # `flow` warps each neighbour onto the target first; `median` does not (the control).
        warp = (mode == "flow")
        tgt = cv2.imread(files[seed_frame])
        g_t = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
        probs = [S.prob(tgt)]
        ph, pw = probs[0].shape[:2]
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if warp else None
        for d in range(-span, span + 1):
            j = seed_frame + d
            if d == 0 or j < 0 or j >= len(files):
                continue
            nb = cv2.imread(files[j])
            if nb is None:
                continue
            p_n = S.prob(nb)
            if not warp:                                        # control: no motion compensation
                probs.append(p_n)
                continue
            g_n = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)
            flow = dis.calc(g_n, g_t, None)                     # neighbour -> target
            # flow is at grey resolution; map it onto the low-res prob grid
            fs = cv2.resize(flow, (pw, ph), interpolation=cv2.INTER_LINEAR)
            sx, sy = pw / g_t.shape[1], ph / g_t.shape[0]
            gx, gy = np.meshgrid(np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32))
            mapx = (gx + fs[..., 0] * sx).astype(np.float32)
            mapy = (gy + fs[..., 1] * sy).astype(np.float32)
            probs.append(cv2.remap(p_n, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        return np.median(np.stack(probs, 0), axis=0), {"ok": True, "mode": mode,
                                                       "n_used": len(probs), "align_err": err}
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def prob_to_mask(prob, shape, largest_blob=None):
    H, W = shape[:2]
    m = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
    if largest_blob is not None and m.any():
        m = largest_blob(m)
    return m
