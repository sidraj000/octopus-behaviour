"""Human-in-the-loop octopus mask labeler (FastAPI, port 8015) — the reliable GT path.

Auto-labeling failed on this footage: GroundingDINO grabs cloth/pipes, CLIP says "octopus"
everywhere, motion catches TVs/people. Every automatic localizer has a distractor. So put the
human exactly where the machines fail (LOCATING the octopus) and keep the machine where it's
strong (SAM2 making + propagating the mask from a correct point).

Per clip:
  1. PRE-SEED with the motion guess (largest motion blob box) -> SAM2 -> mask shown on the seed frame.
  2. If wrong (TV/person/empty), you CLICK the octopus (left = keep / right = exclude) -> SAM2 re-masks.
  3. Accept (A) -> propagate the mask through the clip -> save N clean (image, mask) pairs.
     Reject (R) -> skip (no octopus / unusable).  ←/→ navigate.

Output: data/dataset_seg_human/{images,masks,manifest.jsonl} — trustworthy GT for train + a real val set.
Resumable (skips clips already in the manifest). SAM2 on MPS/CUDA/CPU (auto).

Run:  venv/bin/python3 ui/seg_label.py   ->  http://localhost:8015
"""
import base64, glob, io, json, os, sys, tempfile, subprocess, threading
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from auto_segment import motion_seed, largest_blob, build_prompts

CLIPS_ROOT = Path(os.environ.get("SEG_LABEL_CLIPS", REPO / "src" / "octopus_clips_verified"))
CAMERAS = ["Right_Front", "Right_Back", "Right_Right"]      # colour dens; not Right_Left/Right_Top
OUT = REPO / "data" / "dataset_seg_human"
(OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "masks").mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.jsonl"
CORRUPT_FILE = OUT / "_corrupt.txt"        # clips ffmpeg can't read (truncated / moov atom not found)
FPS = 2; MAXSIDE = 1024; N_PER_CLIP = 4    # fps=2 is enough for the motion pre-seed; faster load
_CORRUPT = set(l.strip() for l in open(CORRUPT_FILE)) if CORRUPT_FILE.exists() else set()
AREA_MIN, AREA_MAX = 0.0008, 0.6

app = FastAPI()
_LOCK = threading.Lock()
_SAM = None       # video predictor (slow init_state) — only used on ACCEPT to propagate
_IMG = None       # image predictor (encodes ONE frame) — used for browse/click, so nav is fast
CUR = {}   # live per-clip state


def _dev():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def img_sam():
    """Fast single-frame predictor for interactive browse/click (no whole-clip encode)."""
    global _IMG
    if _IMG is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _IMG = (SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device=_dev()), _dev())
        print("SAM2 image predictor loaded on", _dev(), flush=True)
    return _IMG


def sam():
    """Video predictor for propagation on ACCEPT only (init_state encodes the whole clip = slow)."""
    global _SAM
    if _SAM is None:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        _SAM = (SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-small", device=_dev()), _dev())
        print("SAM2 video predictor loaded on", _dev(), flush=True)
    return _SAM


def _img_mask(box, points, labels):
    """Predict a single-frame mask on the already-set seed image (fast). CUR['imgs'][seed] must be set()."""
    ip, _ = img_sam()
    kw = {}
    if points:
        kw["point_coords"] = np.array(points, np.float32); kw["point_labels"] = np.array(labels, np.int32)
    if box is not None:
        kw["box"] = np.array(box, np.float32)
    if not kw:
        return np.zeros(CUR["imgs"][CUR["seed_idx"]].size[::-1], bool)
    masks, scores, _ = ip.predict(multimask_output=False, **kw)
    m = masks[0].astype(bool)
    return largest_blob(m) if m.any() else m


def camera_of(p):
    for c in CAMERAS:
        if c in p:
            return c
    return None


def all_clips():
    return sorted(p for p in glob.glob(f"{CLIPS_ROOT}/**/*.mp4", recursive=True)
                  if camera_of(p) and p not in _CORRUPT)


def mark_corrupt(clip):
    if clip not in _CORRUPT:
        _CORRUPT.add(clip)
        with open(CORRUPT_FILE, "a") as f:
            f.write(clip + "\n")


def done_set():
    d = set()
    if MANIFEST.exists():
        for l in open(MANIFEST):
            try:
                d.add(json.loads(l)["clip"])
            except Exception:
                pass
    return d


def source_video(clip):
    p = Path(clip); return f"{p.parent.parent.name}/{p.parent.name}"


def _drop_clip_from_manifest(clip):
    """Remove any existing manifest rows for `clip` (so re-labeling overwrites instead of duplicating)."""
    if not MANIFEST.exists():
        return
    rows = [l for l in open(MANIFEST) if l.strip()]
    keep = [l for l in rows if json.loads(l).get("clip") != clip]
    if len(keep) != len(rows):
        with open(MANIFEST, "w") as f:
            f.writelines(keep)


def _composite_b64(img, mask, points, labels):
    im = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR).copy()
    if mask is not None and mask.any():
        im[mask] = (0.5 * im[mask] + 0.5 * np.array([0, 235, 120])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 0, 255), 2)
    for (x, y), l in zip(points, labels):
        c = (0, 220, 0) if l == 1 else (0, 0, 235)
        cv2.circle(im, (int(x), int(y)), 7, c, -1); cv2.circle(im, (int(x), int(y)), 7, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def load_clip(index):
    clips = all_clips()
    clip = clips[index]
    # clean up previous
    if CUR.get("td"):
        try: CUR["td"].cleanup()
        except Exception: pass
    td = tempfile.TemporaryDirectory()
    fdir = f"{td.name}/f"; os.makedirs(fdir)
    subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vf",
                    f"fps={FPS},scale='min({MAXSIDE},iw)':-2", f"{fdir}/%05d.jpg"], check=False)
    files = sorted(glob.glob(f"{fdir}/*.jpg"))
    imgs = [Image.open(f).convert("RGB") for f in files]
    if not imgs:                          # ffmpeg got no frames (corrupt/unreadable clip) -> skip forever
        td.cleanup(); mark_corrupt(clip); return None
    ms = motion_seed(imgs) if imgs else None
    seed_idx = ms[0] if ms else (len(imgs) // 2 if imgs else 0)
    box = ms[1] if ms else None
    CUR.clear()
    CUR.update(dict(clip=clip, index=index, td=td, fdir=fdir, imgs=imgs,
                    seed_idx=seed_idx, box=box, points=[], labels=[], W=imgs[0].size[0], H=imgs[0].size[1]))
    ip, _ = img_sam()
    ip.set_image(np.asarray(imgs[seed_idx]))          # encode ONE frame (fast) — no whole-clip init
    mask = _img_mask(box, [], []) if imgs else None
    CUR["mask"] = mask
    return {"index": index, "clip": clip, "camera": camera_of(clip), "W": CUR["W"], "H": CUR["H"],
            "seed_frac": None if ms is None else round((box[2]-box[0])*(box[3]-box[1])/(CUR["W"]*CUR["H"]), 3),
            "area": round(float(mask.mean()), 4) if mask is not None else 0.0,
            "img": _composite_b64(imgs[seed_idx], mask, [], [])}


@app.get("/api/state")
def api_state():
    clips = all_clips()
    pos = neg = 0; vset = set()
    if MANIFEST.exists():
        for l in open(MANIFEST):
            try:
                r = json.loads(l)
                if r.get("source") == "human":
                    pos += 1
                elif r.get("source") == "negative":
                    neg += 1
                vset.add(source_video(r["clip"]))
            except Exception:
                pass
    return {"total": len(clips), "saved": pos + neg, "pos": pos, "neg": neg,
            "videos": len(vset), "clips_root": str(CLIPS_ROOT)}


@app.post("/api/load")
def api_load(body: dict):
    with _LOCK:
        idx = int(body.get("index", 0))
        clips = all_clips()
        if not clips:
            return JSONResponse({"error": f"no clips under {CLIPS_ROOT}"}, status_code=404)
        idx = max(0, min(idx, len(clips) - 1))
        step = 1 if int(body.get("dir", 1)) >= 0 else -1
        r = None
        for _ in range(min(400, len(clips))):    # skip broken/unreadable clips in the direction of travel
            if idx < 0 or idx >= len(clips):
                break
            r = load_clip(idx)
            if r is not None:
                break
            idx += step
        if r is None:
            return JSONResponse({"error": "no loadable clip nearby"}, status_code=404)
        r["is_done"] = clips[r["index"]] in done_set()
        return r


@app.post("/api/click")
def api_click(body: dict):
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip loaded"}, status_code=400)
        CUR["points"].append([float(body["x"]), float(body["y"])])
        CUR["labels"].append(int(body.get("label", 1)))
        # clicks REFINE the pre-seed: keep the motion box (if any) and add the points. If the box was
        # wrong (TV/person) the user hits Z first to clear it, then clicks the octopus fresh.
        mask = _img_mask(CUR["box"], CUR["points"], CUR["labels"])
        CUR["mask"] = mask
        return {"area": round(float(mask.mean()), 4),
                "img": _composite_b64(CUR["imgs"][CUR["seed_idx"]], mask, CUR["points"], CUR["labels"])}


@app.post("/api/reset")
def api_reset(body: dict):
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip"}, status_code=400)
        # clear EVERYTHING (drop the motion box + points) -> blank, so the user clicks the octopus fresh
        CUR["points"], CUR["labels"], CUR["box"] = [], [], None
        h, w = CUR["H"], CUR["W"]; mask = np.zeros((h, w), bool); CUR["mask"] = mask
        return {"area": 0.0, "img": _composite_b64(CUR["imgs"][CUR["seed_idx"]], mask, [], [])}


@app.post("/api/accept")
def api_accept(body: dict):
    """Save the HUMAN-VERIFIED seed frame + mask (instant — no whole-clip propagation, which would
    only reintroduce drift). One trustworthy pair per clip; run scripts/propagate_accepted.py later
    if more frames per clip are wanted."""
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip"}, status_code=400)
        mask = CUR.get("mask")
        if mask is None or not mask.any():
            return {"saved": 0, "reason": "empty"}     # nothing to save (skip silently on auto-advance)
        clip = CUR["clip"]; vid = source_video(clip).replace("/", "_"); cam = camera_of(clip)
        if clip in done_set():                          # already labeled -> overwrite its row's files, dedupe manifest
            _drop_clip_from_manifest(clip)
        stem = f"{vid}_{Path(clip).stem}_{cam}_0"
        CUR["imgs"][CUR["seed_idx"]].save(OUT / "images" / f"{stem}.jpg", quality=90)
        Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "masks" / f"{stem}.png")
        with open(MANIFEST, "a") as mf:
            mf.write(json.dumps({"clip": clip, "camera": cam, "image": f"images/{stem}.jpg",
                                 "mask": f"masks/{stem}.png", "area": round(float(mask.mean()), 4),
                                 "seed_frame": CUR["seed_idx"], "n_frames": len(CUR["imgs"]),
                                 "points": CUR["points"], "labels": CUR["labels"], "box": CUR["box"],
                                 "source": "human"}) + "\n")
        return {"saved": 1}


@app.get("/api/resume")
def api_resume():
    """Index to open on page load: the first UNlabeled clip after the last one you reviewed
    (so a reload continues where you left off instead of restarting at 0)."""
    clips = all_clips(); done = done_set()
    last = -1
    for i, c in enumerate(clips):
        if c in done:
            last = i
    nxt = last + 1
    while nxt < len(clips) and clips[nxt] in done:
        nxt += 1
    return {"index": min(nxt, len(clips) - 1) if clips else 0, "last_reviewed": last}


@app.get("/api/next_new")
def api_next_new(after: int = -1):
    """Index of the next clip from a source video that has NO saved/rejected rows yet (for diversity —
    avoids labeling many near-duplicate clips of the same recording)."""
    clips = all_clips()
    labeled_vids = set()
    if MANIFEST.exists():
        for l in open(MANIFEST):
            try:
                labeled_vids.add(source_video(json.loads(l)["clip"]))
            except Exception:
                pass
    for i in range(max(0, after + 1), len(clips)):
        if source_video(clips[i]) not in labeled_vids:
            return {"index": i}
    for i in range(0, len(clips)):     # wrap
        if source_video(clips[i]) not in labeled_vids:
            return {"index": i}
    return {"index": min(after + 1, len(clips) - 1), "all_done": True}


@app.post("/api/negative")
def api_negative(body: dict):
    """Save the current frame as a NO-OCTOPUS negative (frame + all-empty mask). These train the
    presence gate ('no octopus -> empty mask'); earlier this took presence AUC 0.50 -> 0.86."""
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip"}, status_code=400)
        clip = CUR["clip"]; vid = source_video(clip).replace("/", "_"); cam = camera_of(clip)
        if clip in done_set():
            _drop_clip_from_manifest(clip)
        stem = f"{vid}_{Path(clip).stem}_{cam}_neg"
        CUR["imgs"][CUR["seed_idx"]].save(OUT / "images" / f"{stem}.jpg", quality=90)
        Image.fromarray(np.zeros((CUR["H"], CUR["W"]), np.uint8)).save(OUT / "masks" / f"{stem}.png")
        with open(MANIFEST, "a") as mf:
            mf.write(json.dumps({"clip": clip, "camera": cam, "image": f"images/{stem}.jpg",
                                 "mask": f"masks/{stem}.png", "area": 0.0,
                                 "seed_frame": CUR["seed_idx"], "source": "negative"}) + "\n")
        return {"saved": 1}


@app.post("/api/reject")
def api_reject(body: dict):
    with _LOCK:
        if CUR:
            if CUR["clip"] in done_set():
                _drop_clip_from_manifest(CUR["clip"])
            with open(MANIFEST, "a") as mf:
                mf.write(json.dumps({"clip": CUR["clip"], "camera": camera_of(CUR["clip"]),
                                     "image": None, "mask": None, "area": 0.0, "source": "reject"}) + "\n")
        return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Octopus mask labeler</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;flex-direction:column;height:100vh}
 #bar{padding:8px 12px;background:#1b1b1b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #bar b{color:#8f8} .k{color:#888}
 #wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
 #cv{max-width:98%;max-height:98%;cursor:crosshair;border:1px solid #333}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
 button:hover{background:#333} .hint{color:#777;font-size:12px}
 #msg{color:#6cf}
</style></head><body>
<div id=bar>
 <b>Octopus mask labeler</b>
 <span>clip <span id=idx>-</span>/<span id=tot>-</span></span>
 <span class=k>saved</span> <b id=done>-</b> <span class=k>(</span><span id=pos>-</span> <span class=k>octopus +</span> <span id=neg>-</span> <span class=k>empty) from</span> <b id=vids>-</b> <span class=k>videos</span>
 <span id=cam class=k></span> <span>area <span id=area>-</span></span>
 <span id=msg></span>
 <span style="margin-left:auto" class=hint>left-click octopus · right-click exclude · → save&next · <b>N = no-octopus (save negative)</b> · R skip · J jump-new-video · Z clear · ← back</span>
</div>
<div id=wrap><img id=cv></div>
<script>
let idx=0, tot=0, W=1, H=1, busy=false;
const cv=document.getElementById('cv');
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json();}
function setImg(d){ if(d.img) cv.src=d.img; if(d.area!==undefined) document.getElementById('area').textContent=(d.area*100).toFixed(1)+'%'; }
async function refreshState(){const s=await (await fetch('/api/state')).json(); tot=s.total; document.getElementById('tot').textContent=tot; document.getElementById('done').textContent=s.saved; document.getElementById('vids').textContent=s.videos; document.getElementById('pos').textContent=s.pos; document.getElementById('neg').textContent=s.neg;}
async function load(i,dir){ busy=true; msg('loading…'); const d=await post('/api/load',{index:i,dir:(dir||1)}); busy=false;
  if(d.error){msg(d.error);return;} idx=d.index; W=d.W; H=d.H; document.getElementById('idx').textContent=idx+1;
  document.getElementById('cam').textContent=d.camera+(d.is_done?' ✓done':''); setImg(d); msg(d.is_done?'already labeled (re-doing overwrites)':''); refreshState();}
function msg(t){document.getElementById('msg').textContent=t;}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',async e=>{ if(busy)return; e.preventDefault();
  const r=cv.getBoundingClientRect(); const x=(e.clientX-r.left)/r.width*W; const y=(e.clientY-r.top)/r.height*H;
  const label=e.button===2?0:1; busy=true; const d=await post('/api/click',{x,y,label}); busy=false; setImg(d);});
async function saveAndNext(){ busy=true; const d=await post('/api/accept',{}); busy=false;
  if(!d.saved){ msg('⚠ empty mask — CLICK the octopus, or R to skip (nothing saved)'); return; }  // don't advance -> no silent loss
  msg('saved ✓'); await refreshState(); if(idx<tot-1) await load(idx+1); }
async function jumpNew(){ busy=true; const r=await (await fetch('/api/next_new?after='+idx)).json(); busy=false;
  if(r.all_done){msg('all videos have at least one label 🎉');} await load(r.index); msg('jumped to a new video'); }
document.addEventListener('keydown',async e=>{ if(busy)return;
  if(e.key==='ArrowRight'||e.key==='a'||e.key==='A'){await saveAndNext();}
  else if(e.key==='n'||e.key==='N'){busy=true;const d=await post('/api/negative',{});busy=false;msg('saved NO-octopus negative ✓');await refreshState();if(idx<tot-1)await load(idx+1);}
  else if(e.key==='r'||e.key==='R'){busy=true;await post('/api/reject',{});busy=false;msg('skipped (not saved)');await refreshState();if(idx<tot-1)await load(idx+1);}
  else if(e.key==='j'||e.key==='J'){await jumpNew();}
  else if(e.key==='z'||e.key==='Z'){busy=true;const d=await post('/api/reset',{});busy=false;setImg(d);msg('cleared — click the octopus');}
  else if(e.key==='ArrowLeft'){load(Math.max(idx-1,0),-1);}});
(async()=>{await refreshState(); const r=await (await fetch('/api/resume')).json(); await load(r.index); if(r.last_reviewed>=0) msg('resumed after clip '+(r.last_reviewed+1));})();
</script></body></html>"""

if __name__ == "__main__":
    print("Octopus mask labeler -> http://localhost:8015")
    uvicorn.run(app, host="127.0.0.1", port=8015)
