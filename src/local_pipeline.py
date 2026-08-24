"""
Local octopus clip + caption pipeline (video in -> clips + captions out), optimized.

Same result as `local_video_to_captions.ipynb` / `extract_octopus_clips.py` (same gates,
same 20 s non-overlapping clips, same training-matched top-N CLAHE frames), but faster:

  A) SCAN passes run CONCURRENTLY — octopus detection (CLIP on the GPU) and motion
     detection (ffmpeg + numpy on the CPU) overlap instead of running back-to-back.
  B) CAPTIONING REUSES the scan's per-second p_visible — it no longer re-extracts dense
     frames and re-runs CLIP per clip just to pick the best frames. It picks the best-N
     seconds straight from the scan scores and only enhances those frames for the VLM.

Importable (the UI drives `process_video(..., on_stage=, on_clip=)`) and runnable as a CLI:
    python3 local_pipeline.py /path/to/video.mp4 [--camera Right_Top]
"""
import argparse, json, subprocess, sys, tempfile, time, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from caption_openrouter import load_detector, enhance, N_KEEP, IMG_MAXSIDE, PRESENT_MIN

# repo root + default MLX caption student location (bundled in src/ if packaged, else repo models/)
REPO = HERE.parent
_MLX_CANDS = [HERE / "qwen3vl2b_caption_v1_mlx_4bit", REPO / "models" / "qwen3vl2b_caption_v1_mlx_4bit"]
DEFAULT_MLX = next((p for p in _MLX_CANDS if p.exists()), _MLX_CANDS[-1])

# ── pipeline params (defaults match extract_octopus_clips.py) ────────────────────
SAMPLE_FPS       = 1.0
CLIP_LEN         = 20
MIN_VISIBLE_FRAC = 0.50
VIS_THRESH       = 0.60
MOTION_THRESH    = 0.008
MOTION_PIX       = 25
SIZE, BATCH      = 224, 64
CAP_PROMPT = ("These frames are from one short aquarium clip of Nity, an octopus, in time order. "
              "Describe in ONE sentence what the octopus is doing.")


# ── models (loaded once, reused across videos) ──────────────────────────────────
_MODELS = None

def load_models(mlx_model_path=DEFAULT_MLX):
    """Load the CLIP+MLP detector and the MLX caption student once; cached module-globally."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    from mlx_vlm import load as mlx_load
    cm, pre, clf, vis, dev = load_detector()
    mlx_model, mlx_proc = mlx_load(str(mlx_model_path))
    _MODELS = {"cm": cm, "pre": pre, "clf": clf, "vis": vis, "dev": dev,
               "mlx": mlx_model, "proc": mlx_proc}
    return _MODELS


# ── scan: ONE decode feeds both octopus + motion  [SPEEDUP A] ────────────────────
# The old pipeline decoded the whole video twice (once for CLIP, once for motion). Video
# decode is CPU-bound and already multi-threaded, so running the two decodes *concurrently*
# just oversubscribes the cores and is slower. Instead we decode ONCE: ffmpeg does the heavy
# downscale to a fit-224 frame, then per frame Python cheaply (a) pads it to 224² for CLIP and
# (b) stretches it to 224² grey for the motion diff — same geometry as the two original passes.
import cv2

def _probe_scaled_size(path):
    """Native W,H via ffprobe, and the fit-inside-224 size ffmpeg will emit (aspect preserved)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                         capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in out.split("x")[:2])
    s = SIZE / max(w, h)
    return w, h, max(1, round(w * s)), max(1, round(h * s))


def scan_video(path, M, on_stage=None):
    """[A] Single decode → per-second p_visible (CLIP) AND absolute motion, in one pass."""
    if on_stage: on_stage("scanning", "single decode → octopus + motion")
    _, _, sw, sh = _probe_scaled_size(path)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path),
           "-vf", f"fps={SAMPLE_FPS},scale={sw}:{sh}",   # ffmpeg does the expensive downscale once
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fsize = sw * sh * 3
    cm, pre, clf, vis, dev = M["cm"], M["pre"], M["clf"], M["vis"], M["dev"]
    y0, x0 = (SIZE - sh) // 2, (SIZE - sw) // 2
    mrow, mcol = int(SIZE * 0.88), int(SIZE * 0.60)     # burned-in timestamp mask (as in scan_motion_area)
    pv, motion, buf = [], [], []
    prev_g = None

    def flush():
        if not buf: return
        batch = torch.stack([pre(im) for im in buf]).to(dev)
        with torch.no_grad():
            f = cm.encode_image(batch).float(); f = f / f.norm(dim=-1, keepdim=True)
            p = torch.softmax(clf(f), dim=1)[:, vis]
        pv.extend(p.cpu().tolist()); buf.clear()

    while True:
        raw = proc.stdout.read(fsize)
        if len(raw) < fsize: break
        arr = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
        # (a) octopus: pad the fit-224 frame to 224² (== letterbox, no extra resize)
        cv_img = np.full((SIZE, SIZE, 3), 128, np.uint8); cv_img[y0:y0+sh, x0:x0+sw] = arr
        buf.append(Image.fromarray(cv_img))
        # (b) motion: stretch to 224² grey, absolute changed-pixel fraction with timestamp masked
        g = cv2.cvtColor(cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev_g is None:
            motion.append(0.0)
        else:
            diff = np.abs(g - prev_g); diff[mrow:, mcol:] = 0.0
            motion.append(float((diff > MOTION_PIX).mean()))
        prev_g = g
        if len(buf) >= BATCH: flush()
    flush(); proc.stdout.close(); proc.wait()
    return np.array(pv, np.float32), np.array(motion, np.float32)


# ── windows + extraction ────────────────────────────────────────────────────────

def find_windows(pv, motion):
    L = int(CLIP_LEN * SAMPLE_FPS); N = len(pv); out = []; s = 0
    while s + L <= N:
        wp, wm = pv[s:s + L], motion[s:s + L]
        vf = float((wp >= VIS_THRESH).mean()); mm = float(wm.mean())
        if vf > MIN_VISIBLE_FRAC and mm >= MOTION_THRESH:
            out.append({"start": int(s / SAMPLE_FPS), "end": int((s + L) / SAMPLE_FPS),
                        "visible_frac": round(vf, 3), "mean_motion": round(mm, 5)})
            s += L                        # non-overlapping
        else:
            s += 1
    return out


def extract_clip(path, start, end, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return True
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-to", str(end),
                        "-i", str(path), "-c:v", "copy", "-c:a", "aac", str(out_path)],
                       capture_output=True, text=True)
    return r.returncode == 0 and out_path.exists()


# ── caption: reuse scan scores to pick frames  [SPEEDUP B] ───────────────────────

def _extract_frames_at_768(clip_path, tmp):
    """One ffmpeg decode of the (short) clip -> 1 fps frames at <=768px. No CLIP here."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_path),
                    "-vf", f"fps={SAMPLE_FPS},scale='min({IMG_MAXSIDE},iw)':-2", "-q:v", "3",
                    f"{tmp}/f_%03d.jpg"], capture_output=True)
    return sorted(str(p) for p in Path(tmp).glob("f_*.jpg"))


def caption_window(video_path, start, pv, M):
    """[B, no-clip variant] Caption a window WITHOUT keeping a clip file: byte-copy the 20 s window
    to a temp mp4 (robust — same cut the save_clips path uses; deep per-frame seeks into the full
    video fail near truncated stream ends), caption it, then discard the temp clip."""
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "w.mp4"
        if not extract_clip(video_path, start, start + CLIP_LEN, cp):
            return {"caption": None, "status": "noframes"}
        return caption_clip(cp, start, pv, M)


def caption_clip(clip_path, start, pv, M):
    """[B] Pick the best-N seconds from the whole-video scan scores (no per-clip CLIP re-run),
    CLAHE-enhance just those frames, and caption with the MLX student."""
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    win = pv[start:start + CLIP_LEN]
    maxp = float(win.max()) if len(win) else 0.0
    if maxp < PRESENT_MIN:                                   # presence gate (skip the VLM)
        return {"caption": "octopus not present", "max_p_visible": round(maxp, 3), "status": "absent"}
    with tempfile.TemporaryDirectory() as tmp:
        frames = _extract_frames_at_768(clip_path, tmp)
        if not frames:
            return {"caption": None, "status": "noframes"}
        n = min(len(frames), len(win))
        scores = win[:n]
        order = sorted(range(n), key=lambda k: scores[k], reverse=True)[:N_KEEP]
        best = [frames[k] for k in sorted(order)]            # chronological
        prepped = []
        for j, f in enumerate(best):                         # CLAHE == training / teacher input
            im = Image.open(f).convert("RGB"); im.thumbnail((IMG_MAXSIDE, IMG_MAXSIDE)); im = enhance(im)
            outp = f"{tmp}/best_{j:02d}.jpg"; im.save(outp, quality=90); prepped.append(outp)
        fmt = apply_chat_template(M["proc"], M["mlx"].config, CAP_PROMPT, num_images=len(prepped))
        out = mlx_generate(M["mlx"], M["proc"], fmt, prepped, max_tokens=80, temperature=0.0, verbose=False)
        cap = (out.text if hasattr(out, "text") else out).strip()
        return {"caption": cap, "max_p_visible": round(maxp, 3), "status": "captioned"}


# ── orchestration ────────────────────────────────────────────────────────────────

def process_video(video_path, out_dir, M=None, camera="cam", on_stage=None, on_clip=None,
                  save_clips=True):
    """Full pipeline. Calls on_stage(stage, detail) and on_clip(i, total, record) for the UI.
    save_clips=False captions straight from the full video and writes no clip mp4s (the demo UI
    seeks the full video, so per-clip files aren't needed)."""
    M = M or load_models()
    video_path = str(video_path); out_dir = Path(out_dir)
    stem = Path(video_path).stem
    clips_dir = out_dir / "clips"
    if save_clips: clips_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pv, motion = scan_video(video_path, M, on_stage)
    if on_stage: on_stage("scanned", f"{len(pv)}s | mean p_visible {pv.mean():.2f} | "
                                      f"motion {motion.mean():.4f} | {time.time()-t0:.0f}s")

    windows = find_windows(pv, motion)
    if on_stage: on_stage("windows", f"{len(windows)} clips pass both gates")

    recs = []
    for i, w in enumerate(windows, 1):
        rec = {**w, "video_timeline": f"{w['start']//60:02d}:{w['start']%60:02d}-"
                                      f"{w['end']//60:02d}:{w['end']%60:02d}"}
        if save_clips:
            cp = clips_dir / f"{camera}_{stem}_{w['start']:04d}-{w['end']:04d}.mp4"
            if not extract_clip(video_path, w["start"], w["end"], cp):
                continue
            rec["clip_path"] = str(cp); rec["clip_name"] = cp.name
            rec.update(caption_clip(cp, w["start"], pv, M))
        else:
            rec.update(caption_window(video_path, w["start"], pv, M))
        recs.append(rec)
        if on_clip: on_clip(i, len(windows), rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"video": video_path, "camera": camera,
              "processed_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "caption_model": "qwen3vl2b_caption_v1_mlx_4bit",
              "elapsed_sec": round(time.time() - t0, 1),
              "params": {"clip_len": CLIP_LEN, "vis_thresh": VIS_THRESH,
                         "min_visible_frac": MIN_VISIBLE_FRAC, "motion_thresh": MOTION_THRESH},
              "n_clips": len(recs), "clips": recs}
    out_json = out_dir / f"{stem}_captions.json"
    json.dump(result, open(out_json, "w"), indent=2)
    if on_stage: on_stage("done", f"{len(recs)} clips in {result['elapsed_sec']:.0f}s -> {out_json.name}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--out", default=str(REPO / "local_pipeline_out"))
    ap.add_argument("--mlx", default=str(DEFAULT_MLX))
    args = ap.parse_args()
    M = load_models(args.mlx)
    def stage(s, d): print(f"[{s}] {d}", flush=True)
    def clip(i, n, r): print(f"  [{i}/{n}] {r['video_timeline']} ({r['status']}) {r['caption']}", flush=True)
    res = process_video(args.video, args.out, M, camera=args.camera, on_stage=stage, on_clip=clip)
    print(f"\nDONE: {res['n_clips']} clips in {res['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
