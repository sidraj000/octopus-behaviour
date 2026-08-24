"""
Caption + ethogram-classify octopus clips via the OpenRouter API (runs LOCALLY).

Uses the large Qwen3-VL model on OpenRouter (qwen/qwen3-vl-235b-a22b-instruct) — no
GPU / Colab needed, just an API key. For each clip in octopus_clips_verified.json:
  1. sample frames, CLAHE-enhance the dim IR footage
  2. score frames with clip_mlp_hardneg_v2 (p_visible); skip clips with no octopus
  3. send the top-N clearest frames to Qwen3-VL and get ONE caption + one ethogram
     label (7-class ethogram_list_v2.json), or "octopus not present" / "uncertain"
Writes caption + ethogram_label back into the JSON. Resumable; saves after each clip.

Setup: put OPENROUTER_API_KEY in src/.env (or the environment). `ffmpeg` on PATH.
Run:   python3 caption_openrouter.py            # all uncaptioned clips
       python3 caption_openrouter.py --limit 5  # first 5
"""
import os, sys, json, base64, argparse, subprocess, tempfile, time, datetime, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ── config ────────────────────────────────────────────────────────────────────
OR_MODEL     = "qwen/qwen3-vl-235b-a22b-instruct"     # large Qwen3-VL on OpenRouter
OR_URL       = "https://openrouter.ai/api/v1/chat/completions"
INDEX_JSON   = HERE / "octopus_clips_verified.json"
CLIPS_ROOT   = HERE / "octopus_clips_verified"
ETHOGRAM     = HERE / "ethogram_list_v2.json"
CLIP_CKPT    = HERE / "clip_mlp_hardneg_v2.pt"

DENSE_FPS    = 1.0        # candidate frames/sec
N_KEEP       = 6          # best frames sent to the VLM
IMG_MAXSIDE  = 768        # downscale frames sent to the API
PRESENT_MIN  = 0.5        # if no frame's p_visible >= this -> "octopus not present" (skip API)
MAX_TOKENS   = 220
PIPELINE_TAG = "openrouter-qwen3vl235b"


def load_env(p: Path):
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Precedence: real environment > repo-root .env > src/.env.
# `load_env` uses setdefault, so whatever is loaded FIRST wins. The repo root is the canonical
# location for credentials (see README); src/.env exists only so a bare copy of src/ runs standalone.
# Loading src/.env first was a silent trap: a stale key there shadowed a freshly-updated root .env
# and every API call failed with 401 "User not found" while the new key tested fine by hand.
load_env(HERE.parent / ".env")
load_env(HERE / ".env")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ── octopus detector (for best-frame selection + presence gate) ────────────────
def letterbox(img, size=224, fill=(128, 128, 128)):
    w, h = img.size; s = size / max(w, h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    cv = Image.new("RGB", (size, size), fill); cv.paste(img, ((size - nw) // 2, (size - nh) // 2)); return cv

def load_detector():
    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CLIP_CKPT, map_location=dev)
    cm, pre = clip_lib.load(ck["clip_model"], device=dev); cm.eval()
    feat = ck["feat_dim"]; hid = [int(x) for x in ck["arch"].replace("mlp_", "").split("_")]; dims = [feat] + hid + [2]
    L = []
    for i in range(len(dims) - 1):
        L.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2: L += [nn.ReLU(), nn.Dropout(0.3)]
    clf = nn.Sequential(*L).to(dev); clf.load_state_dict(ck["state_dict"]); clf.eval()
    vis = ck.get("label_map", {}).get("visible", 1)
    return cm, pre, clf, vis, dev


def enhance(img):
    """CLAHE on the L channel so the dim IR octopus is visible."""
    try:
        import cv2
        a = np.array(img.convert("RGB")); lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        l, A, B = cv2.split(lab); l = cv2.createCLAHE(2.5, (8, 8)).apply(l)
        return Image.fromarray(cv2.cvtColor(cv2.merge((l, A, B)), cv2.COLOR_LAB2RGB))
    except Exception:
        return img


def extract_frames(clip_path, tmp):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_path),
                    "-vf", f"fps={DENSE_FPS},scale='min(1280,iw)':-2", "-q:v", "3", f"{tmp}/f_%03d.jpg"],
                   capture_output=True)
    return sorted(str(p) for p in Path(tmp).glob("f_*.jpg"))


def score(paths, cm, pre, clf, vis, dev):
    ps = []
    for i in range(0, len(paths), 64):
        batch = [pre(letterbox(Image.open(p).convert("RGB"))) for p in paths[i:i + 64]]
        with torch.no_grad():
            f = cm.encode_image(torch.stack(batch).to(dev)).float(); f = f / f.norm(dim=-1, keepdim=True)
            ps.extend(torch.softmax(clf(f), 1)[:, vis].cpu().tolist())
    return ps


def b64_image(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((IMG_MAXSIDE, IMG_MAXSIDE))
    im = enhance(im)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
        im.save(t.name, quality=88); data = Path(t.name).read_bytes(); Path(t.name).unlink()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


# ── ethogram prompt + parse ────────────────────────────────────────────────────
ethogram = json.load(open(ETHOGRAM))
labels = [b["label"] for b in ethogram["behaviors"]]
valid = set(labels)
label_block = "\n".join(f"- {b['label']}: {b['description']}" for b in ethogram["behaviors"])

def build_prompt():
    return (
        "These are the clearest frames from one short aquarium clip of Nity, an octopus, in time order. "
        "Describe only what you can see.\n"
        "If no octopus is visible, respond EXACTLY:\nCAPTION: octopus not present\nETHOGRAM: octopus not present\n"
        "Otherwise write ONE caption of what the octopus does across the clip, then pick the single best "
        "behavior label below (or 'uncertain' if genuinely unclear):\n"
        f"{label_block}\n\n"
        "Respond EXACTLY:\nCAPTION: <one sentence>\nETHOGRAM: <one label verbatim, or 'uncertain', or 'octopus not present'>"
    )

def parse(text):
    cap, etho = "", None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("CAPTION:"): cap = s[8:].strip().strip("'\"")
        elif s.upper().startswith("ETHOGRAM:"):
            raw = s[9:].strip().strip("'\""); rl = raw.lower()
            if rl in ("uncertain", "unknown"): etho = "uncertain"
            elif "not present" in rl: etho = "octopus not present"
            else:
                for l in valid:
                    if l.lower() == rl or l.lower() in rl or rl in l.lower(): etho = l; break
                else: etho = raw or "uncertain"
    if not cap: cap = text.strip().strip("'\"")
    if "not present" in cap.lower(): return "octopus not present", "octopus not present"
    return cap, (etho or "uncertain")


REQUEST_TIMEOUT = 120     # seconds; callers may raise this (the ensemble does -- see below)


def call_openrouter(image_urls, prompt, retries=4):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    content.append({"type": "text", "text": prompt})
    body = {"model": OR_MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
               "X-Title": "octopus-clip-captioner"}
    for attempt in range(retries):
        r = requests.post(OR_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt); continue
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
    raise RuntimeError("OpenRouter: retries exhausted")


_model_lock = threading.Lock()      # serialize CLIP inference (shared model, MPS not thread-safe)
_json_lock  = threading.Lock()      # serialize index writes


def process_one(e, clips_root, cap_key, etho_key, model_key, prompt, detector):
    """Caption one clip (thread worker). Mutates e in place; returns (e, status).
    status: captioned | absent | missing | noframes | apifail:<msg>"""
    cm, pre, clf, vis, dev = detector
    rel = e["clip_path"].split("octopus_clips_verified/", 1)[-1]
    cp = clips_root / rel
    if not cp.exists():
        return e, "missing"
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(cp, tmp)
        if not frames:
            return e, "noframes"
        with _model_lock:                                   # CLIP scoring is fast; only this is serialized
            sc = score(frames, cm, pre, clf, vis, dev)
        maxp = max(sc)
        e[f"{cap_key}_max_p"] = round(maxp, 4)
        if maxp < PRESENT_MIN:
            e[cap_key] = "octopus not present"; e[etho_key] = "octopus not present"
            status = "absent"
        else:
            order = sorted(range(len(frames)), key=lambda k: sc[k], reverse=True)[:N_KEEP]
            best = [frames[k] for k in sorted(order)]
            imgs = [b64_image(f) for f in best]
            try:
                raw = call_openrouter(imgs, prompt)          # I/O-bound -> the win from running many in parallel
            except Exception as ex:
                return e, f"apifail:{ex}"
            cap, etho = parse(raw)
            e[cap_key] = cap; e[etho_key] = etho
            status = "captioned"
    e[model_key] = OR_MODEL
    e[f"{cap_key}_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return e, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max clips to caption this run")
    ap.add_argument("--workers", type=int, default=10, help="concurrent API requests (I/O-bound -> big speedup)")
    ap.add_argument("--index", type=str, default=str(INDEX_JSON), help="clip-index JSON to read/write")
    ap.add_argument("--clips-root", type=str, default=str(CLIPS_ROOT), help="dir the clip mp4s live under")
    ap.add_argument("--cap-key", type=str, default="caption",
                    help="field to write the caption into (e.g. caption_235b to compare vs an existing caption)")
    ap.add_argument("--etho-key", type=str, default="ethogram_label", help="field to write the ethogram label into")
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("OPENROUTER_API_KEY not set — put it in src/.env or the environment.")

    index_path = Path(args.index); clips_root = Path(args.clips_root)
    cap_key, etho_key = args.cap_key, args.etho_key
    model_key = f"{cap_key}_model"

    prompt = build_prompt()
    index = json.load(open(index_path)); clips = index["clips"]
    todo = [c for c in clips if not c.get(cap_key)]           # resume on THIS caption field
    if args.limit: todo = todo[:args.limit]
    print(f"{len(clips)} clips | {len(todo)} to caption -> '{cap_key}' via {OR_MODEL} | {args.workers} workers\n" + "-" * 60, flush=True)
    if not todo:
        print("nothing to do."); return

    detector = load_detector()
    done = absent = apifail = missing = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, e, clips_root, cap_key, etho_key, model_key, prompt, detector): e for e in todo}
        for fut in as_completed(futs):
            e, status = fut.result()
            completed += 1
            if status == "captioned":   done += 1
            elif status == "absent":    absent += 1; done += 1
            elif status == "missing":   missing += 1
            elif status.startswith("apifail"): apifail += 1
            label = e.get(etho_key, status); cap = str(e.get(cap_key, ""))[:70]
            print(f"[{completed}/{len(todo)}] {status} :: {label} :: {cap}", flush=True)
            with _json_lock:                                 # periodic durable save (resumable); full save at end
                if completed % 20 == 0:
                    json.dump(index, open(index_path, "w"), indent=2)
    with _json_lock:
        json.dump(index, open(index_path, "w"), indent=2)

    print("-" * 60 + f"\nDone. captioned {done} ({absent} auto/vlm no-octopus, {apifail} api-fail, {missing} missing-file). "
          f"'{cap_key}' filled: {sum(1 for c in clips if c.get(cap_key))}/{len(clips)}")


if __name__ == "__main__":
    main()
