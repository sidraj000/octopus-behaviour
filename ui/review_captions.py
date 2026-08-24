"""
Caption review UI — watch each extracted clip (from local storage) next to its
Qwen caption + ethogram label, and approve / reject / correct it.

Reads data/octopus_clips_verified.json (entries already have `caption` and
`ethogram_label`). Decisions are written BACK into that same JSON per clip:
  - review        : "approved" | "rejected"
  - caption        : editable (overwrites if you change it)
  - ethogram_label : editable (dropdown of ethogram_list.json + "octopus not present")
  - reviewed_at    : timestamp
Saved after every action, so it is safe to close and resume.

Videos are streamed from local disk (data/octopus_clips_verified/...), no copy.

Usage:  venv/bin/python3 ui/review_captions.py   ->  http://localhost:8005
"""
import json, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

PROJECT   = Path(__file__).resolve().parent.parent
INDEX     = PROJECT / "data" / "octopus_clips_verified.json"
CLIPS_DIR = PROJECT / "data" / "octopus_clips_verified"
ETHOGRAM  = PROJECT / "data" / "ethogram_list_v2.json"   # compact 7-behavior sheet

app = FastAPI()


def load_index() -> dict:
    return json.load(open(INDEX))


def save_index(idx: dict):
    idx["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump(idx, open(INDEX, "w"), indent=2)


def labels() -> list:
    try:
        return [b["label"] for b in json.load(open(ETHOGRAM))["behaviors"]] + ["octopus not present"]
    except Exception:
        return ["octopus not present"]


@app.get("/video")
def video(path: str):
    # path is the entry's clip_path (repo-relative). Only serve files under CLIPS_DIR.
    p = (PROJECT / path).resolve()
    if not str(p).startswith(str(CLIPS_DIR.resolve())) or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")


@app.post("/save")
async def save(req: Request):
    body = await req.json()
    idx = load_index()
    n_app = n_rej = 0
    for c in idx["clips"]:
        if c["clip_path"] == body["clip_path"]:
            if "review" in body:
                if body["review"] == "clear":
                    c.pop("review", None)
                else:
                    c["review"] = body["review"]
            if "caption" in body:
                c["caption"] = body["caption"]
            if "ethogram_label" in body:
                c["ethogram_label"] = body["ethogram_label"]
            c["reviewed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    for c in idx["clips"]:
        if c.get("review") == "approved":
            n_app += 1
        elif c.get("review") == "rejected":
            n_rej += 1
    save_index(idx)
    return JSONResponse({"ok": True, "approved": n_app, "rejected": n_rej})


@app.get("/", response_class=HTMLResponse)
def index():
    idx = load_index()
    clips = idx["clips"]
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Caption review</title><style>
 *{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0d0d0d;color:#eee;
   height:100vh;display:flex;flex-direction:column;overflow:hidden}}
 header{{background:#1c1c1c;padding:8px 16px;border-bottom:1px solid #333;display:flex;
   align-items:center;gap:18px;flex-wrap:wrap}}
 h1{{font-size:15px;margin:0}} .counts span{{margin-right:12px;font-size:13px}}
 .keys{{font-size:12px;color:#aaa;margin-left:auto}} .keys b{{color:#ddd}}
 .stage{{flex:1;display:flex;min-height:0;gap:14px;padding:14px}}
 .vid{{flex:1.4;display:flex;align-items:center;justify-content:center;min-width:0}}
 video{{max-width:100%;max-height:100%;border:4px solid #333;border-radius:8px;background:#000}}
 video.approved{{border-color:#2ecc71}} video.rejected{{border-color:#e74c3c}}
 .panel{{flex:1;display:flex;flex-direction:column;gap:12px;overflow:auto}}
 .meta span{{display:inline-block;background:#262626;border-radius:5px;padding:3px 8px;margin:2px;font-size:12px}}
 label{{font-size:12px;color:#9ab;display:block;margin-bottom:4px}}
 textarea{{width:100%;background:#161616;color:#eee;border:1px solid #333;border-radius:6px;
   padding:8px;font-size:15px;line-height:1.4;resize:vertical;min-height:90px}}
 select{{width:100%;background:#161616;color:#eee;border:1px solid #333;border-radius:6px;padding:8px;font-size:14px}}
 .path{{font-size:11px;color:#888;word-break:break-all}}
 .verdict{{font-weight:700;font-size:15px}} .v-approved{{color:#2ecc71}} .v-rejected{{color:#e74c3c}}
 footer{{background:#1c1c1c;border-top:1px solid #333;padding:8px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap}}
 footer button{{border:0;padding:9px 16px;border-radius:6px;color:#fff;cursor:pointer;font-size:14px}}
 .ok{{background:#27632a}} .bad{{background:#7b241c}} .nav{{background:#222}} .save{{background:#2c3e50}}
 footer button:hover{{filter:brightness(1.3)}}
</style></head><body>
<header><h1>🐙 Caption review</h1>
<div class="counts"><span>clip <b id=pos>1</b>/<b id=tot>0</b></span>
<span style="color:#2ecc71">✓ <b id=app>0</b></span>
<span style="color:#e74c3c">✗ <b id=rej>0</b></span>
<span>left <b id=un>0</b></span></div>
<div class="keys"><b>A</b> approve · <b>R</b> reject · <b>→</b>/<b>Space</b> next · <b>←</b> back · <b>U</b> clear · <b>S</b> save edits</div>
</header>
<div class="stage">
  <div class="vid"><video id="vid" controls autoplay loop muted></video></div>
  <div class="panel">
    <div class="meta" id="meta"></div>
    <div><label>Caption <span id="dirtyflag" style="color:#e0b000"></span></label><textarea id="cap" oninput="dirty=true;document.getElementById('dirtyflag').textContent='editing…'" onblur="persistCurrent()"></textarea></div>
    <div><label>Ethogram label</label><select id="etho" onchange="dirty=true;persistCurrent()"></select></div>
    <div class="verdict" id="verdict"></div>
    <div class="path" id="path"></div>
  </div>
</div>
<footer>
 <button class="nav" onclick="go(-1)">← Back</button>
 <button class="bad" onclick="mark('rejected')">✗ Reject (R)</button>
 <button class="ok" onclick="mark('approved')">✓ Approve (A)</button>
 <button class="save" onclick="saveEdits()">💾 Save edits (S)</button>
 <button class="nav" onclick="go(1)">Next →</button>
</footer>
<script>
const clips={json.dumps(clips)};
const LABELS={json.dumps(labels())};
let i=0;
let dirty=false;
const $=id=>document.getElementById(id);
// ethogram dropdown options
$('etho').innerHTML=LABELS.map(l=>`<option value="${{l}}">${{l}}</option>`).join('');
// start at first unreviewed
i=clips.findIndex(c=>!c.review); if(i<0)i=0;
function counts(){{
 const a=clips.filter(c=>c.review=='approved').length;
 const r=clips.filter(c=>c.review=='rejected').length;
 $('app').textContent=a; $('rej').textContent=r;
 $('un').textContent=clips.length-a-r; $('tot').textContent=clips.length;
}}
function fmt(v){{return v===undefined||v===null?'–':v;}}
function render(){{
 const c=clips[i];
 $('vid').src='/video?path='+encodeURIComponent(c.clip_path);
 $('vid').className=c.review||'';
 $('meta').innerHTML=
   `<span>${{c.camera}}</span><span>${{c.date}} ${{c.segment}}</span>`+
   `<span>⏱ ${{fmt(c.video_timeline)}}</span>`+
   `<span>vis ${{fmt(c.visible_frac)}}</span><span>motion ${{fmt(c.mean_motion)}}</span>`+
   `<span>p ${{fmt(c.mean_p)}}</span><span>src ${{fmt(c.source)}}</span>`;
 $('cap').value=c.caption||'';
 $('etho').value=LABELS.includes(c.ethogram_label)?c.ethogram_label:'octopus not present';
 $('path').textContent=c.clip_path;
 $('pos').textContent=i+1;
 const v=c.review;
 $('verdict').className='verdict'+(v?' v-'+v:'');
 $('verdict').textContent=v=='approved'?'✓ APPROVED':v=='rejected'?'✗ REJECTED':'';
 dirty=false; $('dirtyflag').textContent='';
 counts();
}}
async function post(body){{
 const r=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify(body)}});
 return r.json();
}}
function captureCurrent(){{ const c=clips[i]; c.caption=$('cap').value; c.ethogram_label=$('etho').value; return c; }}
function persistCurrent(){{                       // auto-save pending caption/label edits
 if(!dirty) return;
 const c=captureCurrent();
 post({{clip_path:c.clip_path,caption:c.caption,ethogram_label:c.ethogram_label}});
 dirty=false; $('dirtyflag').textContent='saved ✓';
}}
function go(d){{ persistCurrent(); i=Math.max(0,Math.min(clips.length-1,i+d)); render(); }}
function mark(v){{
 const c=clips[i];
 if(c.review==v){{ delete c.review; post({{clip_path:c.clip_path,review:'clear'}}); render(); return; }}
 c.review=v;
 // also persist any edits currently in the fields
 c.caption=$('cap').value; c.ethogram_label=$('etho').value;
 post({{clip_path:c.clip_path,review:v,caption:c.caption,ethogram_label:c.ethogram_label}});
 render(); setTimeout(()=>go(1),150);
}}
function saveEdits(){{
 const c=clips[i];
 c.caption=$('cap').value; c.ethogram_label=$('etho').value;
 post({{clip_path:c.clip_path,caption:c.caption,ethogram_label:c.ethogram_label}});
 $('verdict').textContent='💾 saved'; setTimeout(render,600);
}}
function clearOne(){{ const c=clips[i]; delete c.review; post({{clip_path:c.clip_path,review:'clear'}}); render(); }}
document.addEventListener('keydown',e=>{{
 if(e.target.tagName=='TEXTAREA'||e.target.tagName=='SELECT') return;  // don't hijack typing
 if(e.key=='a'||e.key=='A')mark('approved');
 else if(e.key=='r'||e.key=='R')mark('rejected');
 else if(e.key==' '||e.key=='ArrowRight'){{e.preventDefault();go(1);}}
 else if(e.key=='ArrowLeft')go(-1);
 else if(e.key=='u'||e.key=='U')clearOne();
 else if(e.key=='s'||e.key=='S')saveEdits();
}});
render();
</script></body></html>"""


if __name__ == "__main__":
    print("Caption review → http://localhost:8005")
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="warning")
