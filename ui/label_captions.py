"""
Blind caption labeling UI — build the caption training set.

Per clip (played from local disk) you see TWO candidate captions, "Option A" and
"Option B" — one is v1, one is v2, shuffled per clip and NEVER revealed. Click one
to adopt it (it fills the editable textarea), or ignore both and write your own.
Pick the ethogram label, hit Submit.

Each submission is appended to data/caption_training_set.json — a NEW file, for
training later. The server (not the browser) knows the A/B mapping and records
where the final caption came from: "v1", "v2", or "human" (edited/written), so we
can analyze pipeline quality later without ever un-blinding the review itself.

Resumable: already-submitted clips are skipped on load; saved after every submit.

Usage:  venv/bin/python3 ui/label_captions.py   ->  http://localhost:8008
"""
import json, hashlib, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

PROJECT   = Path(__file__).resolve().parent.parent
V1_JSON   = PROJECT / "data" / "octopus_clips_verified.json"
V2_JSON   = PROJECT / "data" / "octopus_clips_verified-2.json"
CLIPS_DIR = PROJECT / "data" / "octopus_clips_verified"
ETHOGRAM  = PROJECT / "data" / "ethogram_list_v2.json"   # compact 7-behavior sheet
OUT_JSON  = PROJECT / "data" / "caption_training_set.json"

app = FastAPI()


def a_is_v1(clip_path: str) -> bool:
    """Stable per-clip shuffle: whether Option A is the v1 caption."""
    return int(hashlib.md5(clip_path.encode()).hexdigest(), 16) % 2 == 0


def load_sources():
    v1 = {c["clip_path"]: c for c in json.load(open(V1_JSON))["clips"]}
    v2 = {c["clip_path"]: c for c in json.load(open(V2_JSON))["clips"]}
    return v1, v2


def _remap():
    """old-label -> compact-label, from the v2 sheet's maps_from."""
    m = {}
    try:
        for bd in json.load(open(ETHOGRAM))["behaviors"]:
            for old in bd.get("maps_from", []):
                m[old] = bd["label"]
            m[bd["label"]] = bd["label"]
    except Exception:
        pass
    m["octopus not present"] = "octopus not present"
    return m

REMAP = _remap()

def remap_label(lab):
    return REMAP.get(lab, lab) if lab else ""


def load_items():
    """One item per clip that exists locally. Captions + labels sent as anonymous A/B."""
    v1, v2 = load_sources()
    items = []
    for cp, a in v1.items():
        if not (PROJECT / cp).exists() or cp not in v2:
            continue
        b = v2[cp]
        cap1, cap2 = a.get("caption", "") or "", b.get("caption", "") or ""
        lab1, lab2 = remap_label(a.get("ethogram_label")), remap_label(b.get("ethogram_label"))
        if a_is_v1(cp):
            optA, optB, labA, labB = cap1, cap2, lab1, lab2
        else:
            optA, optB, labA, labB = cap2, cap1, lab2, lab1
        items.append({
            "clip_path": cp, "camera": a.get("camera"), "date": a.get("date"),
            "segment": a.get("segment"), "video_timeline": a.get("video_timeline"),
            "optA": optA, "optB": optB, "optA_label": labA, "optB_label": labB,
        })
    return items


def load_out():
    if OUT_JSON.exists():
        return json.load(open(OUT_JSON))
    return {"description": "Human-curated caption training set: blind-picked (v1/v2) or "
                           "hand-written caption + ethogram label per clip.",
            "updated_at": None, "count": 0, "entries": []}


def labels() -> list:
    try:
        return [b["label"] for b in json.load(open(ETHOGRAM))["behaviors"]] + ["octopus not present"]
    except Exception:
        return ["octopus not present"]


@app.get("/video")
def video(path: str):
    p = (PROJECT / path).resolve()
    if not str(p).startswith(str(CLIPS_DIR.resolve())) or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")


@app.post("/submit")
async def submit(req: Request):
    body = await req.json()
    cp = body["clip_path"]
    caption = (body.get("caption") or "").strip()
    label = body.get("ethogram_label")
    if not caption or not label:
        return JSONResponse({"error": "caption and ethogram_label required"}, status_code=400)

    # derive the source server-side (the browser never knows which was v1/v2)
    v1, v2 = load_sources()
    c1 = (v1.get(cp, {}).get("caption") or "").strip()
    c2 = (v2.get(cp, {}).get("caption") or "").strip()
    source = "v1" if caption == c1 else "v2" if caption == c2 else "human"

    meta = v1.get(cp, {})
    out = load_out()
    out["entries"] = [e for e in out["entries"] if e["clip_path"] != cp]  # resubmit replaces
    out["entries"].append({
        "clip_path": cp, "camera": meta.get("camera"), "date": meta.get("date"),
        "segment": meta.get("segment"), "video_timeline": meta.get("video_timeline"),
        "video_url": meta.get("video_url"),
        "caption": caption, "ethogram_label": label, "caption_source": source,
        "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    out["entries"].sort(key=lambda e: e["clip_path"])
    out["count"] = len(out["entries"])
    out["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    return JSONResponse({"ok": True, "count": out["count"]})


@app.get("/", response_class=HTMLResponse)
def index():
    items = load_items()
    done = {e["clip_path"]: {"caption": e["caption"], "ethogram_label": e["ethogram_label"]}
            for e in load_out()["entries"]}
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Caption labeling (blind)</title><style>
 *{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0d0d0d;color:#eee;
   height:100vh;display:flex;flex-direction:column;overflow:hidden}}
 header{{background:#1c1c1c;padding:8px 16px;border-bottom:1px solid #333;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
 h1{{font-size:15px;margin:0}} .counts span{{margin-right:10px;font-size:13px}}
 .keys{{font-size:12px;color:#aaa;margin-left:auto}} .keys b{{color:#ddd}}
 .stage{{flex:1;display:flex;min-height:0;gap:14px;padding:14px}}
 .vid{{flex:1;display:flex;align-items:center;justify-content:center;min-width:0}}
 video{{max-width:100%;max-height:100%;border:3px solid #333;border-radius:8px;background:#000}}
 video.done{{border-color:#2ecc71}}
 .panel{{flex:1.2;display:flex;flex-direction:column;gap:10px;overflow:auto}}
 .meta span{{display:inline-block;background:#262626;border-radius:5px;padding:3px 8px;margin:2px;font-size:12px}}
 .opt{{background:#161616;border:2px solid #333;border-radius:8px;padding:10px;cursor:pointer}}
 .opt:hover{{border-color:#557}} .opt.picked{{border-color:#4a9eff;background:#12203a}}
 .opt h3{{margin:0 0 6px;font-size:12px;color:#89a;letter-spacing:.5px}}
 .opt .cap{{font-size:14px;line-height:1.45}}
 .olab{{margin-top:7px;font-size:12px;color:#cdb;background:#243024;display:inline-block;padding:2px 8px;border-radius:5px}}
 label{{font-size:12px;color:#9ab;display:block;margin-bottom:4px}}
 textarea{{width:100%;background:#161616;color:#eee;border:1px solid #333;border-radius:6px;
   padding:8px;font-size:14px;line-height:1.4;resize:vertical;min-height:80px}}
 select{{width:100%;background:#161616;color:#eee;border:1px solid #333;border-radius:6px;padding:8px;font-size:14px}}
 .path{{font-size:11px;color:#888;word-break:break-all}}
 .status{{font-weight:700;font-size:14px;min-height:18px}} .ok{{color:#2ecc71}} .err{{color:#e74c3c}}
 footer{{background:#1c1c1c;border-top:1px solid #333;padding:8px;display:flex;gap:10px;justify-content:center}}
 footer button{{border:0;padding:9px 18px;border-radius:6px;color:#fff;cursor:pointer;font-size:14px}}
 .nav{{background:#222}} .sub{{background:#27632a;font-weight:700}} footer button:hover{{filter:brightness(1.3)}}
</style></head><body>
<header><h1>✍️ Caption labeling <span style="color:#789;font-size:12px">(blind A/B — pick, edit, or write your own)</span></h1>
<div class="counts"><span>clip <b id=pos>1</b>/<b id=tot>0</b></span>
<span style="color:#2ecc71">submitted <b id=nsub>0</b></span><span>left <b id=nleft>0</b></span></div>
<div class="keys"><b>1</b> option A · <b>2</b> option B · <b>Enter</b>(outside text) submit · <b>←/→</b> nav</div>
</header>
<div class="stage">
  <div class="vid"><video id="vid" controls autoplay loop muted></video></div>
  <div class="panel">
    <div class="meta" id="meta"></div>
    <div class="opt" id="optA" onclick="pick('A')"><h3>OPTION A</h3><div class="cap" id="capA"></div><div class="olab" id="lA"></div></div>
    <div class="opt" id="optB" onclick="pick('B')"><h3>OPTION B</h3><div class="cap" id="capB"></div><div class="olab" id="lB"></div></div>
    <div><label>Final caption (edit freely, or write your own)</label>
      <textarea id="cap" oninput="unpickIfEdited()"></textarea></div>
    <div><label>Ethogram label</label><select id="etho"></select></div>
    <div class="status" id="status"></div>
    <div class="path" id="path"></div>
  </div>
</div>
<footer>
 <button class="nav" onclick="go(-1)">← Back</button>
 <button class="sub" onclick="submit()">✔ Submit &amp; next</button>
 <button class="nav" onclick="go(1)">Skip →</button>
</footer>
<script>
const items={json.dumps(items)};
let done={json.dumps(done)};
const LABELS={json.dumps(labels())};
let i=0, picked=null;
const $=id=>document.getElementById(id);
$('etho').innerHTML='<option value="">— choose label —</option>'+LABELS.map(l=>`<option value="${{l}}">${{l}}</option>`).join('');
i=items.findIndex(it=>!done[it.clip_path]); if(i<0)i=0;
function counts(){{
 $('nsub').textContent=Object.keys(done).length;
 $('nleft').textContent=items.length-Object.keys(done).length;
 $('tot').textContent=items.length;
}}
function fmt(v){{return v===undefined||v===null?'–':v;}}
function render(){{
 const it=items[i]; picked=null;
 $('vid').src='/video?path='+encodeURIComponent(it.clip_path);
 $('meta').innerHTML=`<span>${{it.camera}}</span><span>${{it.date}} ${{it.segment}}</span><span>⏱ ${{fmt(it.video_timeline)}}</span>`;
 $('capA').textContent=it.optA||'(empty)';
 $('capB').textContent=it.optB||'(empty)';
 $('lA').textContent=it.optA_label?'label: '+it.optA_label:'';
 $('lB').textContent=it.optB_label?'label: '+it.optB_label:'';
 $('optA').classList.remove('picked'); $('optB').classList.remove('picked');
 const d=done[it.clip_path];
 $('cap').value=d?d.caption:'';
 $('etho').value=d?d.ethogram_label:'';
 $('vid').className=d?'done':'';
 $('status').textContent=d?'✓ already submitted (resubmit to replace)':'';
 $('status').className='status ok';
 $('path').textContent=it.clip_path;
 $('pos').textContent=i+1;
 counts();
}}
function pick(which){{
 const it=items[i]; picked=which;
 $('cap').value=which=='A'?it.optA:it.optB;
 const lab=which=='A'?it.optA_label:it.optB_label;      // adopt that option's label too
 if(LABELS.includes(lab)) $('etho').value=lab;
 $('optA').classList.toggle('picked',which=='A');
 $('optB').classList.toggle('picked',which=='B');
}}
function unpickIfEdited(){{
 const it=items[i];
 if($('cap').value!==it.optA && $('cap').value!==it.optB){{
   picked=null; $('optA').classList.remove('picked'); $('optB').classList.remove('picked');
 }}
}}
function go(d){{ i=Math.max(0,Math.min(items.length-1,i+d)); render(); }}
async function submit(){{
 const it=items[i];
 const caption=$('cap').value.trim(), label=$('etho').value;
 if(!caption){{ $('status').textContent='caption is empty'; $('status').className='status err'; return; }}
 if(!label){{ $('status').textContent='choose an ethogram label'; $('status').className='status err'; return; }}
 const r=await fetch('/submit',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{clip_path:it.clip_path,caption:caption,ethogram_label:label}})}});
 if(r.ok){{
   done[it.clip_path]={{caption:caption,ethogram_label:label}};
   $('status').textContent='✓ saved'; $('status').className='status ok';
   counts(); setTimeout(()=>{{ const n=items.findIndex((x,k)=>k>i&&!done[x.clip_path]);
     i=n>=0?n:Math.min(items.length-1,i+1); render(); }},250);
 }} else {{
   $('status').textContent='save failed'; $('status').className='status err';
 }}
}}
document.addEventListener('keydown',e=>{{
 if(e.target.tagName=='TEXTAREA'||e.target.tagName=='SELECT') return;
 if(e.key=='1')pick('A');
 else if(e.key=='2')pick('B');
 else if(e.key=='Enter')submit();
 else if(e.key=='ArrowRight'){{e.preventDefault();go(1);}}
 else if(e.key=='ArrowLeft')go(-1);
}});
render();
</script></body></html>"""


if __name__ == "__main__":
    print("Blind caption labeling → http://localhost:8008")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="warning")
