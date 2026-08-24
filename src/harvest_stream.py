"""harvest_stream.py — diverse-footage harvester (Phase A/B), streaming + early-exit.

Crawls the footage server's HTML directory listings, and for each colour-camera video:
  stream 1fps via ffmpeg HTTP -> per-second p_visible (CLIP+MLP) + absolute motion ->
  slide non-overlapping 20s windows -> keep a window when >50% frames visible (p>=0.6)
  AND mean motion >= thresh -> STOP at CLIPS_PER_VIDEO (early-exit) -> extract those
  windows with ffmpeg byte-range copy straight from the URL. Never downloads full videos.

Goal = VIDEO/scene diversity: sample a few segments per (date,camera) across MANY days,
2 clips each, rather than exhausting any one day.

Tracking (resumable): ONE ledger json keyed by video_url — records status + the clips found.
Each clip also emitted in octopus_clips_verified.json entry format for easy merge.

Self-contained: needs torch + openai-clip + numpy + pillow + requests + ffmpeg, and the
detector weight (clip_mlp_hardneg_v2.pt). Device auto cpu/mps/cuda. Runs locally or on Modal.
"""
import argparse, base64, json, os, re, subprocess, sys, threading, time, datetime, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import requests

# ── config / params ──────────────────────────────────────────────────────────────
BASE = "https://repo.octopus-intelligence.org/public/"
NITY_COLLECTIONS = ["O-vulgaris-Nity-2025-9-17--/", "O-vulgaris-Nity-2026-2-20--/"]
# colour den cameras — server naming varies slightly per collection; match case-insensitively
COLOUR_CAMS = ["right front", "right back", "right right"]
SAMPLE_FPS = 1.0
CLIP_LEN = 20
MIN_VISIBLE_FRAC = 0.50
VIS_THRESH = 0.60
MOTION_THRESH = 0.008
MOTION_PIX = 25
REQUIRE_MOTION = False        # data harvest: gate on VISIBILITY only (still-but-visible octopus is
                              # great training data for seg/caption). Motion still recorded as metadata.
SIZE, BATCH = 224, 64
CLIPS_PER_VIDEO = 2            # early-exit after this many good windows
SPREAD_SEC = 60               # min gap between kept window STARTS (within-video variety)
MAX_SEG_PER_DAYCAM = 3        # sample up to this many segments per (date,camera) -> diversity
N_PROBES = 10                 # probe-first: cheap frames sampled across the video before full scan
PROBE_THRESH = 0.50           # if NO probe frame reaches this p_visible -> skip full scan (empty)
MAX_SCAN_SEC = 0              # 0 = scan whole video (early-exit still applies); >0 caps scan

USER = os.environ.get("OCTOPUS_USER", "")
PASS = os.environ.get("OCTOPUS_PASS", "")
AUTH = "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()
HDRS = {"Authorization": AUTH}
_LOCK = threading.Lock()      # guards CLIP inference (single model) + ledger writes


# ── detector (letterbox + CLIP ViT-B/32 + MLP probe) ─────────────────────────────
def letterbox(img, size=SIZE, fill=(128, 128, 128)):
    w, h = img.size; s = size / max(w, h); nw, nh = max(1, round(w*s)), max(1, round(h*s))
    img = img.resize((nw, nh), Image.BICUBIC)
    cv = Image.new("RGB", (size, size), fill); cv.paste(img, ((size-nw)//2, (size-nh)//2)); return cv


def load_detector(ckpt):
    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev)
    cm, pre = clip_lib.load(ck["clip_model"], device=dev); cm.eval()
    feat = ck["feat_dim"]; hid = [int(x) for x in ck["arch"].replace("mlp_", "").split("_")]
    dims = [feat] + hid + [2]; L = []
    for i in range(len(dims)-1):
        L.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2: L += [nn.ReLU(), nn.Dropout(0.3)]
    clf = nn.Sequential(*L).to(dev); clf.load_state_dict(ck["state_dict"]); clf.eval()
    vis = ck.get("label_map", {}).get("visible", 1)
    return {"cm": cm, "pre": pre, "clf": clf, "vis": vis, "dev": dev}


# ── server crawl ─────────────────────────────────────────────────────────────────
def links(url):
    r = requests.get(url, headers=HDRS, timeout=40); r.raise_for_status()
    return re.findall(r'href="([^"?][^"]*)"', r.text)


def list_dates(coll_url):
    """camera-dir -> {date: listing_url}. Handles Local/ subdir."""
    out = {}
    for cam in [l for l in links(coll_url) if l.endswith("/") and l != "../"]:
        if urllib.parse.unquote(cam).strip("/").lower() not in COLOUR_CAMS:
            continue
        for sub in (coll_url + cam + "Local/", coll_url + cam):
            try:
                ds = [l for l in links(sub) if re.match(r"\d{4}-\d{2}-\d{2}/?$", l)]
            except Exception:
                ds = []
            if ds:
                for d in ds:
                    out.setdefault((urllib.parse.unquote(cam).strip("/"), d.strip("/")), sub + d)
                break
    return out


def list_videos(date_url):
    return [date_url.rstrip("/") + "/" + l for l in links(date_url) if l.lower().endswith(".mp4")]


def build_plan(collections, plan_limit=0, date_min=None, date_max=None, reverse=False):
    """-> list of (collection, camera, date, video_url), sampling up to MAX_SEG_PER_DAYCAM
    segments per (date,camera), spread across the day. plan_limit>0 stops the (slow) crawl
    early once that many entries are collected — makes small test runs fast.

    date_min/date_max (inclusive, "YYYY-MM-DD") restrict the plan to a date window, and
    reverse=True walks dates newest-first. Together these let two boxes split ONE collection
    without a shared ledger: give box A `--date-max D` and box B `--date-reverse --date-min D+1`,
    and they work toward each other over disjoint footage. Dates are ISO so string compare
    is chronological.
    """
    plan = []
    for coll in collections:
        cu = BASE + coll
        dates = list_dates(cu)
        items = sorted(dates.items(), key=lambda kv: (kv[0][0], kv[0][1]), reverse=reverse)
        for (cam, date), durl in items:
            if date_min and date < date_min:
                continue
            if date_max and date > date_max:
                continue
            try:
                vids = sorted(list_videos(durl))
            except Exception:
                continue
            if not vids:
                continue
            n = min(MAX_SEG_PER_DAYCAM, len(vids))
            idx = np.linspace(0, len(vids)-1, n).astype(int)   # spread across the day
            for i in idx:
                plan.append((urllib.parse.unquote(coll).strip("/"), cam, date, vids[i]))
            if plan_limit and len(plan) >= plan_limit:
                return plan
    return plan


# ── stream-scan a video (early-exit) ─────────────────────────────────────────────
def scan_stream(url, M):
    """Stream 1fps -> (pv[], motion[]) until CLIPS_PER_VIDEO windows found or video ends.
    ffmpeg letterboxes each frame to exactly SIZE×SIZE (scale-decrease + pad = the training transform)."""
    import cv2
    cmd = ["ffmpeg", "-loglevel", "error", "-headers", f"Authorization: {AUTH}\r\n", "-i", url,
           "-vf", (f"fps={SAMPLE_FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
                   f"pad={SIZE}:{SIZE}:-1:-1:color=gray"),
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cm, pre, clf, vis, dev = M["cm"], M["pre"], M["clf"], M["vis"], M["dev"]
    pv, motion, buf = [], [], []
    prev_g = None
    L = int(CLIP_LEN * SAMPLE_FPS)

    def flush():
        if not buf: return
        with _LOCK:
            batch = torch.stack([pre(im) for im in buf]).to(dev)
            with torch.no_grad():
                f = cm.encode_image(batch).float(); f = f / f.norm(dim=-1, keepdim=True)
                pr = torch.softmax(clf(f), dim=1)[:, vis]
        pv.extend(pr.cpu().tolist()); buf.clear()

    def n_windows():
        w = 0; s = 0
        while s + L <= len(pv) and len(motion) >= s + L:
            wp = np.array(pv[s:s+L]); wm = np.array(motion[s:s+L])
            if (wp >= VIS_THRESH).mean() > MIN_VISIBLE_FRAC and (not REQUIRE_MOTION or wm.mean() >= MOTION_THRESH):
                w += 1; s += max(L, int(SPREAD_SEC * SAMPLE_FPS))   # spread kept windows apart
            else:
                s += 1
        return w

    fsize = SIZE * SIZE * 3
    while True:
        raw = p.stdout.read(fsize)
        if len(raw) < fsize:
            break
        arr = np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)
        buf.append(Image.fromarray(arr))
        g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev_g is None:
            motion.append(0.0)
        else:
            diff = np.abs(g - prev_g); diff[int(SIZE*0.88):, int(SIZE*0.60):] = 0.0
            motion.append(float((diff > MOTION_PIX).mean()))
        prev_g = g
        if len(buf) >= BATCH:
            flush()
            if n_windows() >= CLIPS_PER_VIDEO: break
        if MAX_SCAN_SEC and len(motion) >= MAX_SCAN_SEC: break
    flush()
    p.kill(); p.wait()
    return np.array(pv, np.float32), np.array(motion, np.float32)


def find_windows(pv, motion):
    L = int(CLIP_LEN * SAMPLE_FPS); out = []; s = 0
    while s + L <= len(pv) and len(motion) >= s + L:
        wp, wm = pv[s:s+L], motion[s:s+L]
        vf = float((wp >= VIS_THRESH).mean()); mm = float(wm.mean())
        if vf > MIN_VISIBLE_FRAC and (not REQUIRE_MOTION or mm >= MOTION_THRESH):
            out.append({"start": int(s/SAMPLE_FPS), "end": int((s+L)/SAMPLE_FPS),
                        "visible_frac": round(vf, 3), "mean_motion": round(mm, 5)})
            s += max(L, int(SPREAD_SEC * SAMPLE_FPS))
            if len(out) >= CLIPS_PER_VIDEO: break
        else:
            s += 1
    return out


# ── probe-first: cheaply decide whether a video is worth a full scan ──────────────
def probe_duration(url):
    r = subprocess.run(["ffprobe", "-v", "error", "-headers", f"Authorization: {AUTH}\r\n",
                        "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", url],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _grab_frame(url, t):
    """One letterboxed 224² frame at time t via a fast input-seek (small range download)."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t),
           "-headers", f"Authorization: {AUTH}\r\n", "-i", url, "-frames:v", "1",
           "-vf", (f"scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
                   f"pad={SIZE}:{SIZE}:-1:-1:color=gray"),
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if len(r.stdout) < SIZE * SIZE * 3:
        return None
    return Image.fromarray(np.frombuffer(r.stdout[:SIZE*SIZE*3], np.uint8).reshape(SIZE, SIZE, 3))


def probe_video(url, M, duration, n=N_PROBES):
    """Sample n frames spread across [5%,95%] of the video, return [(t, p_visible), ...].
    Cheap: n small input-seeks instead of streaming the whole file."""
    if duration <= 0:
        return []
    ts = np.linspace(0.05 * duration, 0.95 * duration, n)
    frames, tsk = [], []
    for t in ts:
        im = _grab_frame(url, float(t))
        if im is not None:
            frames.append(im); tsk.append(float(t))
    if not frames:
        return []
    cm, pre, clf, vis, dev = M["cm"], M["pre"], M["clf"], M["vis"], M["dev"]
    with _LOCK:
        batch = torch.stack([pre(im) for im in frames]).to(dev)
        with torch.no_grad():
            f = cm.encode_image(batch).float(); f = f / f.norm(dim=-1, keepdim=True)
            p = torch.softmax(clf(f), dim=1)[:, vis].cpu().tolist()
    return list(zip(tsk, p))


def extract_clip(url, start, end, out_path, timeout=180):
    """Byte-range copy one clip. Video-only, and never leaves a broken file behind.

    THREE bugs were found here on 2026-08-21, all silent:

    1. `-c copy` FAILS ON Right_Right. Those cameras record audio as `pcm_alaw`, which MP4 cannot
       contain: "Could not find tag for codec pcm_alaw in stream #1" -> rc=234, 0-byte output. It had
       produced **929 zero-byte Right_Right clips**. `-c:v copy -an` fixes it (verified: 8.20 MB,
       20.034 s) and we never use the audio anyway.
    2. NO TIMEOUT. One ffmpeg sat blocked on a stalled server connection for 2h11m, which hung the
       whole refetch batch and stalled the download loop waiting on it.
    3. A FAILED RUN LEFT ITS OUTPUT FILE. Downstream code globs for *.mp4 and treats any file as a
       real clip, so 0-byte files were fed to the captioner/ensemble as if they were footage --
       guaranteed failures that looked like API throttling. Now the file is removed on failure so the
       clip is simply missing, and the refetcher will try it again.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return True
    try:
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y",
                            "-headers", f"Authorization: {AUTH}\r\n",
                            "-ss", str(start), "-to", str(end), "-i", url,
                            "-c:v", "copy", "-an", str(out_path)],
                           capture_output=True, timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    ok = rc == 0 and out_path.exists() and out_path.stat().st_size > 10000
    if not ok and out_path.exists():
        try:
            out_path.unlink()          # never leave a stub that downstream globs will trust
        except OSError:
            pass
    return ok


# ── orchestration ────────────────────────────────────────────────────────────────
def process_video(item, M, out_root):
    """Probe-first: cheaply probe N frames; if none look like octopus, skip the full scan.
    The returned record is a DETAILED coverage report — duration, exact probe points, how much
    of the video we scanned, and why it was discarded — so we can mine more later if needed."""
    coll, cam, date, url = item
    seg = re.sub(r"[^0-9]", "", url.split("/")[-1].split("--")[0]) or url.split("/")[-1]
    try:
        dur = probe_duration(url)
        probe = probe_video(url, M, dur)
        pmax = max((p for _, p in probe), default=0.0)
        rec = {"duration": round(dur, 1),
               "probe_points": [[round(t, 1), round(p, 3)] for t, p in probe],
               "probe_max_p": round(pmax, 3), "n_probes": len(probe)}
        # empty by probe -> skip full scan, but record exactly what we covered
        if dur > 0 and probe and pmax < PROBE_THRESH:
            rec.update({"status": "probed_empty", "coverage": "probe_only", "scanned_sec": 0,
                        "unscanned_sec": int(dur), "discard_reason": f"probe_max_p {pmax:.2f} < {PROBE_THRESH}",
                        "n_clips": 0, "clips": []})
            return rec
        # promising (or no duration / probe failed) -> full stream-scan with early-exit
        pv, motion = scan_stream(url, M)
        wins = find_windows(pv, motion)
        clips = []
        for w in wins:
            camtag = cam.replace(" ", "_")
            rel = f"{coll}/{date}/{seg}/{camtag}_{w['start']}-{w['end']}.mp4"
            if extract_clip(url, w["start"], w["end"], Path(out_root) / rel):
                clips.append({"video": url.split("/")[-1], "video_url": url, "collection": coll,
                              "date": date, "segment": seg, "camera": camtag,
                              "start_sec": w["start"], "end_sec": w["end"],
                              "video_timeline": f"{w['start']//60:02d}:{w['start']%60:02d}-{w['end']//60:02d}:{w['end']%60:02d}",
                              "visible_frac": w["visible_frac"], "mean_motion": w["mean_motion"],
                              "clip_path": rel, "added_at": datetime.datetime.now().isoformat(timespec="seconds")})
        scanned = int(len(motion))
        # early-exit means we stopped scanning after the last kept window -> rest is unexplored
        rec.update({"status": "clips" if clips else "scanned_empty",
                    "coverage": f"0-{scanned}s", "scanned_sec": scanned,
                    "unscanned_sec": max(0, int(dur) - scanned) if dur else None,
                    "discard_reason": None if clips else "scanned, no window passed the gate",
                    "n_clips": len(clips), "clips": clips})
        return rec
    except Exception as e:
        return {"status": "failed", "error": f"{type(e).__name__}: {e}", "n_clips": 0, "clips": []}


def run(out, ckpt, collections=NITY_COLLECTIONS, workers=2, max_seg=3, limit=0,
        max_scan_sec=0, commit_cb=None, date_min=None, date_max=None, reverse=False):
    """Core harvest loop. commit_cb() (if given) is called after each ledger save — used on
    Modal to persist the Volume periodically (resumability across timeouts/crashes)."""
    global MAX_SEG_PER_DAYCAM, MAX_SCAN_SEC
    MAX_SEG_PER_DAYCAM = max_seg; MAX_SCAN_SEC = max_scan_sec
    assert USER and PASS, "set OCTOPUS_USER / OCTOPUS_PASS"

    out_root = Path(out); out_root.mkdir(parents=True, exist_ok=True)
    ledger_path = out_root / "harvest_ledger.json"
    index_path = out_root / "harvest_clips_index.json"
    ledger = json.load(open(ledger_path)) if ledger_path.exists() else {}

    print("building plan (crawling server)...", flush=True)
    plan = build_plan(collections, plan_limit=(limit * 4 if limit else 0),
                      date_min=date_min, date_max=date_max, reverse=reverse)
    plan = [it for it in plan if it[3] not in ledger]
    if limit: plan = plan[:limit]
    win = f" [dates {date_min or '..'}..{date_max or '..'}{', newest-first' if reverse else ''}]" \
        if (date_min or date_max or reverse) else ""
    print(f"plan: {len(plan)} videos to scan ({len(ledger)} already in ledger){win}", flush=True)

    M = load_detector(ckpt)
    print(f"detector loaded (dev={M['dev']}).", flush=True)
    done = {"total_clips": 0}   # per-status counts added dynamically; total_clips = sum of n_clips

    def save():
        with _LOCK:
            json.dump(ledger, open(ledger_path, "w"), indent=1)
            idx = [c for e in ledger.values() for c in e.get("clips", [])]
            json.dump({"description": "harvest clips (stream+early-exit)", "clips": idx}, open(index_path, "w"), indent=1)
        if commit_cb: commit_cb()

    def work(it):
        res = process_video(it, M, out_root)
        with _LOCK:
            ledger[it[3]] = {"collection": it[0], "camera": it[1], "date": it[2], **res,
                             "scanned_at": datetime.datetime.now().isoformat(timespec="seconds")}
            done[res["status"]] = done.get(res["status"], 0) + 1
            done["total_clips"] += res["n_clips"]
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(work, plan)):
            if (i + 1) % 10 == 0:
                save()
                print(f"[{i+1}/{len(plan)}] clips={done.get('clips',0)} probed_empty={done.get('probed_empty',0)} "
                      f"scanned_empty={done.get('scanned_empty',0)} total_clips={done['total_clips']}", flush=True)
    save()
    print(f"\nDONE. {done}\n-> {out_root}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="harvest_out")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent / "clip_mlp_hardneg_v2.pt"))
    ap.add_argument("--collections", nargs="+", default=NITY_COLLECTIONS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-seg-per-daycam", type=int, default=MAX_SEG_PER_DAYCAM)
    ap.add_argument("--max-scan-sec", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--date-min", default=None, help="only dates >= this (YYYY-MM-DD, inclusive)")
    ap.add_argument("--date-max", default=None, help="only dates <= this (YYYY-MM-DD, inclusive)")
    ap.add_argument("--date-reverse", action="store_true",
                    help="walk dates newest-first (pair with --date-min to split a collection "
                         "across two boxes working toward each other)")
    args = ap.parse_args()
    run(args.out, args.ckpt, args.collections, args.workers, args.max_seg_per_daycam,
        args.limit, args.max_scan_sec, date_min=args.date_min, date_max=args.date_max,
        reverse=args.date_reverse)


if __name__ == "__main__":
    main()
