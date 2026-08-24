"""
Octopus clip extractor — the clean, consolidated pipeline.

One script that does the three things the clip pipeline actually needs, each the
*correct* way:

  1. OCTOPUS DETECTION — CLIP ViT-B/32 + MLP probe (clip_mlp_hardneg_v2.pt, bundled in src/),
     letterbox preprocessing (no crop), per-second p_visible.
  2. MOTION DETECTION  — scan_motion_area() from motion_detector.py: the ABSOLUTE
     changed-pixel fraction with the burned-in timestamp masked out. (NOT the old
     per-video normalized motion, which lets a flickering lamp pass.)
  3. CLIP EXTRACTION   — slide a 20s non-overlapping window; keep a window when
     >50% of its frames are octopus-visible AND its mean motion clears the
     absolute threshold; ffmpeg byte-range copy of just that window.

This supersedes the exp27 (buggy normalized motion) + exp28 (octopus re-check) +
exp30 (motion re-audit) chain: because both gates are correct here, clips come
out clean in a single pass — no separate cleanup step required.

Per video it makes two 1 fps ffmpeg passes (one for octopus, one for motion via
scan_motion_area) so the motion logic stays a single source of truth.

Self-contained: run this folder anywhere. Needs `.env` (copy `.env.example` -> `.env`,
fill OCTOPUS_USER/OCTOPUS_PASS) or those as env vars; `ffmpeg` on PATH; `pip install -r
requirements.txt`. The detector weight is bundled here.

Outputs (all inside this folder):
  octopus_clips_verified/{date}/{segment}/{Camera}_{start}-{end}.mp4  — clips
  octopus_clips_verified.json     — clip index (source url + time + scores)
  octopus_clips_processed.json    — processed-video ledger (resumable)
  (extract_clip skips a path that already exists, so existing clips are not overwritten)

Usage (from inside src/):
  python3 extract_octopus_clips.py --limit 1
  python3 extract_octopus_clips.py --date 2026-02-20
  python3 extract_octopus_clips.py --motion-thresh 0.01
"""
import argparse, datetime, json, re, subprocess, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from motion_detector import scan_motion_area   # same folder (src/); correct absolute motion

HERE = Path(__file__).resolve().parent            # src/ — fully self-contained
sys.path.insert(0, str(HERE))
from server_creds import USER, PASS               # bundled server_creds.py; reads src/.env or env vars

CKPT_PATH   = HERE / "clip_mlp_hardneg_v2.pt"                     # bundled octopus detector
CLIPS_DIR   = HERE / "octopus_clips_verified"                    # extracted clip mp4s land here
INDEX_JSON  = HERE / "octopus_clips_verified.json"               # clip index — captions filled later by the captioning notebook
PROCESSED   = HERE / "octopus_clips_processed.json"              # processed-video ledger (do-not-reprocess)

BASE = "https://repo.octopus-intelligence.org/public/O-vulgaris-Nity-2026-2-20--"
CAMERAS = ["Right Back", "Right Front", "Right Left", "Right Right", "Right Top"]

# ── pipeline params ───────────────────────────────────────────────────────────
SAMPLE_FPS       = 1.0     # 1 frame/sec for both octopus + motion
CLIP_LEN         = 20      # seconds per clip
MIN_VISIBLE_FRAC = 0.50    # > this fraction of window frames must be octopus-visible
VIS_THRESH       = 0.60    # p_visible >= this -> frame counts as "visible"
MOTION_THRESH    = 0.008   # mean ABSOLUTE changed-pixel fraction in window (raised from 0.005:
                           # 0.005 let IR-noise/reflection false positives through, esp. Right_Left)
MOTION_PIX       = 25      # per-pixel grey-level change counted as "moved"
SIZE, BATCH      = 224, 64


def auth(url: str) -> str:
    return url.replace("https://", f"https://{USER}:{PASS}@")


def letterbox(img, size=224, fill=(128, 128, 128)):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ── server enumeration ──────────────────────────────────────────────────────────

def _curl(url: str) -> str:
    return subprocess.run(["curl", "-s", "--user", f"{USER}:{PASS}", url],
                          capture_output=True, text=True).stdout

def list_dates() -> list:
    out = _curl(f"{BASE}/Right%20Top/Local/")
    return sorted(set(re.findall(r'href="(\d{4}-\d{2}-\d{2})/"', out)))

def list_segments(cam: str, date: str) -> list:
    enc = urllib.parse.quote(cam)
    out = _curl(f"{BASE}/{enc}/Local/{date}/")
    rows = []
    for f in re.findall(r'href="([^"]+\.mp4)"', out):
        m = re.match(r"(\d+)--", f)
        if not m:
            continue
        seg = m.group(1); cam_us = cam.replace(" ", "_")
        rows.append({"video": f"data/aquarium/full/{date}/{seg}/{cam_us}.mp4",
                     "date": date, "segment": seg, "camera": cam_us,
                     "url": f"{BASE}/{enc}/Local/{date}/{f}"})
    return rows

def enumerate_candidates(dates: list, cams: list) -> list:
    tasks = [(c, d) for d in dates for c in cams]
    out = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda a: list_segments(*a), tasks):
            out.extend(r)
    return out


# ── registries ──────────────────────────────────────────────────────────────────

def load_json(path: Path, default):
    return json.load(open(path)) if path.exists() else default

def init_registries():
    proc = load_json(PROCESSED, {
        "task": "octopus_clip_extraction",
        "description": "Videos processed by the consolidated 20s-clip pipeline "
                       "(scan_motion_area motion + >50% octopus visible). Do not reprocess.",
        "updated_at": None, "count": 0, "processed": []})
    idx = load_json(INDEX_JSON, {
        "description": "Extracted 20s octopus clips: source video url + time location + scores.",
        "model": CKPT_PATH.name,
        "updated_at": None, "count": 0, "clips": []})
    return proc, idx

def save_registries(proc, idx):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    proc["count"] = len(proc["processed"]); proc["updated_at"] = now
    idx["count"] = len(idx["clips"]);        idx["updated_at"] = now
    json.dump(proc, open(PROCESSED, "w"), indent=2)
    json.dump(idx,  open(INDEX_JSON, "w"), indent=2)


# ── model ─────────────────────────────────────────────────────────────────────

def build_clf(ck) -> nn.Module:
    feat = ck["feat_dim"]; arch = ck.get("arch", "linear")
    if arch == "linear":
        return nn.Linear(feat, 2)
    hid = [int(x) for x in arch.replace("mlp_", "").split("_")]; dims = [feat] + hid + [2]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(0.3)]
    return nn.Sequential(*layers)

def load_model(device):
    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib
    ck = torch.load(CKPT_PATH, map_location=device)
    cm, pp = clip_lib.load(ck["clip_model"], device=device); cm.eval()
    clf = build_clf(ck).to(device); clf.load_state_dict(ck["state_dict"]); clf.eval()
    vis = ck.get("label_map", {}).get("visible", 1)
    print(f"model: {CKPT_PATH.name}  {ck['clip_model']}+{ck.get('arch')}  acc={ck.get('test_acc',0):.1%}")
    return cm, pp, clf, vis


# ── octopus pass: per-second p_visible (single 1 fps ffmpeg stream) ──────────────

def classify_video(url, cm, pp, clf, vis_idx, device):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", auth(url),
           "-vf", (f"fps={SAMPLE_FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
                   f"pad={SIZE}:{SIZE}:-1:-1:color=gray"),   # letterbox — matches training
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fsize = SIZE * SIZE * 3
    pv, buf = [], []

    def flush():
        if not buf:
            return
        batch = torch.stack([pp(im) for im in buf]).to(device)
        with torch.no_grad():
            f = cm.encode_image(batch).float(); f = f / f.norm(dim=-1, keepdim=True)
            p = torch.softmax(clf(f), dim=1)[:, vis_idx]
        pv.extend(p.cpu().tolist()); buf.clear()

    while True:
        raw = proc.stdout.read(fsize)
        if len(raw) < fsize:
            break
        arr = np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)
        buf.append(letterbox(Image.fromarray(arr)))
        if len(buf) >= BATCH:
            flush()
    flush()
    proc.stdout.close(); proc.wait()
    return np.array(pv, np.float32)


def motion_aligned_to(pv_len: int, url: str) -> np.ndarray:
    """Per-second absolute motion (scan_motion_area), aligned to the pv index grid."""
    ts, scores = scan_motion_area(auth(url), fps=SAMPLE_FPS, pix_thresh=MOTION_PIX)
    m = np.zeros(pv_len, np.float32)
    for k, t in enumerate(ts):
        i = int(round(float(t)))
        if 0 <= i < pv_len:
            m[i] = scores[k]
    return m


# ── window finding + extraction ──────────────────────────────────────────────

def find_clip_windows(pv: np.ndarray, motion: np.ndarray) -> list:
    L = int(CLIP_LEN * SAMPLE_FPS); N = len(pv)
    out, s = [], 0
    while s + L <= N:
        win_pv, win_mo = pv[s:s + L], motion[s:s + L]
        vis_frac = float((win_pv >= VIS_THRESH).mean())
        mean_mo  = float(win_mo.mean())
        if vis_frac > MIN_VISIBLE_FRAC and mean_mo >= MOTION_THRESH:
            out.append({"start_sec": int(s / SAMPLE_FPS), "end_sec": int((s + L) / SAMPLE_FPS),
                        "visible_frac": round(vis_frac, 3), "mean_motion": round(mean_mo, 5)})
            s += L            # non-overlapping
        else:
            s += 1
    return out

def extract_clip(url, start, end, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return True
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-to", str(end),
           "-i", auth(url), "-c:v", "copy", "-c:a", "aac", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out_path.exists()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global MOTION_THRESH, MIN_VISIBLE_FRAC, VIS_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max videos to process this run")
    ap.add_argument("--date", type=str, default=None, help="restrict to a single date YYYY-MM-DD")
    ap.add_argument("--motion-thresh", type=float, default=MOTION_THRESH,
                    help="mean absolute changed-pixel fraction per window (default %(default)s)")
    ap.add_argument("--visible-frac", type=float, default=MIN_VISIBLE_FRAC,
                    help="min fraction of window frames that must be octopus-visible (default %(default)s)")
    ap.add_argument("--vis-thresh", type=float, default=VIS_THRESH,
                    help="p_visible >= this -> frame counts as octopus-visible (default %(default)s)")
    args = ap.parse_args()
    MOTION_THRESH = args.motion_thresh
    MIN_VISIBLE_FRAC = args.visible_frac
    VIS_THRESH = args.vis_thresh

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    cm, pp, clf, vis_idx = load_model(device)
    print(f"device: {device}  | clip {CLIP_LEN}s | p_visible>={VIS_THRESH} "
          f"| visible-frac>{MIN_VISIBLE_FRAC} | motion>={MOTION_THRESH} (pix {MOTION_PIX})")

    proc_reg, clip_idx = init_registries()
    done = {r["video"] for r in proc_reg["processed"]}

    dates = [args.date] if args.date else list_dates()
    cands = enumerate_candidates(dates, CAMERAS)
    todo = [c for c in cands if c["video"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(cands)} candidate videos; {len(todo)} to process\n" + "-" * 64, flush=True)

    for i, c in enumerate(todo, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(todo)}] {c['date']} {c['segment']} {c['camera']}", flush=True)
        try:
            pv = classify_video(c["url"], cm, pp, clf, vis_idx, device)
        except Exception as e:
            print(f"   ! octopus scan failed: {e}"); continue
        if len(pv) == 0:
            print("   ! no frames; skipping"); continue
        try:
            motion = motion_aligned_to(len(pv), c["url"])
        except Exception as e:
            print(f"   ! motion scan failed: {e}; skipping"); continue

        windows = find_clip_windows(pv, motion)
        n_saved = 0
        for w in windows:
            clip_path = (CLIPS_DIR / c["date"] / c["segment"]
                         / f"{c['camera']}_{w['start_sec']:04d}-{w['end_sec']:04d}.mp4")
            if extract_clip(c["url"], w["start_sec"], w["end_sec"], clip_path):
                clip_idx["clips"].append({
                    "video": c["video"], "video_url": c["url"], "date": c["date"],
                    "segment": c["segment"], "camera": c["camera"],
                    "start_sec": w["start_sec"], "end_sec": w["end_sec"],
                    "video_timeline": f"{w['start_sec']//60:02d}:{w['start_sec']%60:02d}-"
                                      f"{w['end_sec']//60:02d}:{w['end_sec']%60:02d}",
                    "visible_frac": w["visible_frac"], "mean_motion": w["mean_motion"],
                    "clip_path": str(clip_path.relative_to(HERE)),
                    "added_at": datetime.datetime.now().isoformat(timespec="seconds")})
                n_saved += 1
        proc_reg["processed"].append({"video": c["video"], "date": c["date"],
            "segment": c["segment"], "camera": c["camera"],
            "n_clips": n_saved, "n_frames": int(len(pv)), "sources": ["extract_octopus_clips"]})
        done.add(c["video"]); save_registries(proc_reg, clip_idx)
        print(f"   {len(pv)} frames | {len(windows)} windows -> {n_saved} clips "
              f"| {time.perf_counter() - t0:.0f}s", flush=True)

    print("-" * 64)
    print(f"DONE. clips: {clip_idx['count']} | processed videos: {proc_reg['count']}")
    print(f"index -> {INDEX_JSON.relative_to(HERE)} | clips dir -> {CLIPS_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
