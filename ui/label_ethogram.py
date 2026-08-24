"""label_ethogram.py — BLIND human ethogram labelling UI (port 8021).

Creates the human ground truth the project has never had: ~900 hand labels exist for masks, presence
negatives and hard negatives, and effectively zero for behaviour -- while every headline behavioural
result is grouped by a behaviour label whose accuracy is unmeasured (R15 gave consistency only, and
the paper names human validation as its open item).

BLIND BY CONSTRUCTION. The model's caption, its ethogram, its vote margin and the clip's stratum are
never sent to the browser. `ui/review_captions.py` shows the model's answer, which yields anchored
approve/reject rather than an independent judgement -- that measures agreement with the model, not
accuracy. Here the payload carries the clip path and nothing else.

Reads  : data/human_eval_sample_v1.json   (frozen sample; _model_* fields stay server-side)
Writes : data/human_behaviour_labels.json (after every action, so it is resumable)

Records per clip: present (yes/no), ethogram (7 classes), confidence (sure/unsure), optional note,
plus seconds spent -- a label given in 2 s is worth knowing about when the analysis runs.

Keys: 1-7 behaviour · 0 no octopus · u unsure-toggle · ArrowLeft/Right prev/next · s skip
Run  : venv/bin/python3 ui/label_ethogram.py  ->  http://localhost:8021
"""
import datetime, json, time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
import os
# switchable so a finished round is never overwritten by the next one
VERSION = os.environ.get("EVAL_VERSION", "v2")
SAMPLE = REPO / "data" / f"human_eval_sample_{VERSION}.json"
OUT = REPO / "data" / (f"human_behaviour_labels.json" if VERSION == "v1"
                       else f"human_behaviour_labels_{VERSION}.json")
ETHO = REPO / "src" / "ethogram_list_v2.json"
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]

app = FastAPI()
LABELS = [b["label"] for b in json.load(open(ETHO))["behaviors"]]


def sample():
    return json.load(open(SAMPLE))["clips"]


def load_out():
    return json.load(open(OUT)) if OUT.exists() else {}


def save_out(d):
    tmp = OUT.with_suffix(".tmp")
    json.dump(d, open(tmp, "w"), indent=1)
    tmp.replace(OUT)


def resolve(clip):
    for r in ROOTS:
        p = r / clip
        if p.exists():
            return p
    return None


@app.get("/api/queue")
def queue():
    """Clip paths ONLY -- no model output, no stratum, nothing that could anchor the labeller."""
    done = load_out()
    items = [{"clip": c["clip"], "camera": c["camera"], "date": c["date"],
              "on_disk": resolve(c["clip"]) is not None} for c in sample()]
    return {"version": VERSION, "labels": LABELS, "n": len(items), "n_done": sum(1 for i in items if i["clip"] in done),
            "items": items, "done": {k: v for k, v in done.items()}}


@app.get("/clip")
def clip(path: str):
    p = resolve(path)
    if not p:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")


_BY_CLIP = {c["clip"]: c for c in sample()}


@app.get("/api/hint")
def hint(clip: str):
    """The ensemble's verdict for ONE clip, only when explicitly requested.

    Deliberately NOT part of /api/queue: seeing the model's answer before judging anchors the
    labeller, which converts an accuracy measurement into an agreement measurement. Kept behind a
    separate call so the client must ask, and so `assisted` (recorded on the label) is meaningful.
    """
    c = _BY_CLIP.get(clip)
    if not c:
        return JSONResponse({"error": "unknown clip"}, status_code=404)
    return {"ethogram": c.get("_model_ethogram"), "present": c.get("_model_present"),
            "votes": c.get("_model_votes"), "margin": c.get("_model_margin"),
            "unanimous": c.get("_model_unanimous"), "low_conf": c.get("_model_low_conf")}


class Rec(BaseModel):
    clip: str
    present: bool | None = None
    ethogram: str | None = None
    unsure: bool = False
    note: str = ""
    seconds: float = 0.0
    skipped: bool = False
    # True if the model's suggestion was on screen when this label was committed. The analysis MUST
    # split on this: assisted labels measure agreement with the model, blind ones measure accuracy.
    assisted: bool = False


@app.post("/api/save")
def save(r: Rec):
    d = load_out()
    d[r.clip] = {"present": r.present, "ethogram": r.ethogram, "unsure": r.unsure,
                 "note": r.note, "seconds": round(r.seconds, 1), "skipped": r.skipped,
                 "assisted": r.assisted,
                 "at": datetime.datetime.now().isoformat(timespec="seconds")}
    save_out(d)
    return {"ok": True, "n_done": len(d)}


PAGE = """
<!doctype html><meta charset=utf-8><title>Ethogram labelling</title>
<style>
 :root{--bg:#12141a;--fg:#e8eaf0;--mut:#9aa3b2;--acc:#4a9eff;--ok:#31c26d;--no:#e5484d;--card:#1b1e26}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{display:flex;align-items:center;gap:16px;padding:10px 18px;background:var(--card);
   border-bottom:1px solid #262b36;position:sticky;top:0;z-index:5}
 #bar{flex:1;height:6px;background:#262b36;border-radius:3px;overflow:hidden}
 #fill{height:100%;background:var(--acc);width:0}
 main{display:grid;grid-template-columns:1fr 320px;gap:18px;padding:18px;max-width:1500px;margin:0 auto}
 video{width:100%;background:#000;border-radius:8px}
 .panel{background:var(--card);border:1px solid #262b36;border-radius:10px;padding:14px}
 button{display:block;width:100%;margin:6px 0;padding:11px 12px;border-radius:8px;cursor:pointer;
   background:#232833;color:var(--fg);border:1px solid #313846;text-align:left;font-size:14px}
 button:hover{border-color:var(--acc)} button kbd{color:var(--mut);float:right;font-size:12px}
 button.sel{background:var(--acc);color:#04101f;border-color:var(--acc);font-weight:600}
 .no{background:#2a1d1f;border-color:#4a2a2c} .no.sel{background:var(--no);color:#fff}
 .row{display:flex;gap:8px} .row button{margin:6px 0}
 textarea{width:100%;background:#0f1116;color:var(--fg);border:1px solid #313846;border-radius:8px;
   padding:8px;font:13px/1.4 inherit;resize:vertical}
 .meta{color:var(--mut);font-size:13px} .warn{color:#f5a524}
 #toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--ok);
   color:#04160c;padding:8px 16px;border-radius:8px;opacity:0;transition:.2s;font-weight:600}
</style>
<header>
  <b>Ethogram labelling</b><span class=meta id=ver></span>
  <span class=meta id=pos></span>
  <div id=bar><div id=fill></div></div>
  <span class=meta id=cnt></span>
</header>
<main>
  <div>
    <video id=v controls autoplay loop muted playsinline></video>
    <p class=meta id=info></p>
  </div>
  <div>
    <div class=panel>
      <div class=meta style="margin-bottom:6px">What is the octopus doing?</div>
      <div id=btns></div>
      <button class=no id=bno>No octopus visible <kbd>0</kbd></button>
    </div>
    <div class=panel style="margin-top:12px">
      <button id=bhint>Show model suggestion <kbd>h</kbd></button>
      <div id=hint style="display:none;margin:8px 0;padding:10px;background:#0f1116;
           border:1px solid #3a3f4d;border-radius:8px;font-size:13px"></div>
      <label class=meta style="display:block;margin:6px 0">
        <input type=checkbox id=always> always show (labels get marked <i>assisted</i>)
      </label>
      <button id=bu>Mark unsure <kbd>u</kbd></button>
      <textarea id=note rows=3 placeholder="note (optional)"></textarea>
      <div class=row>
        <button id=bprev>&larr; Prev</button><button id=bskip>Skip <kbd>s</kbd></button>
      </div>
      <p class=meta>Hidden by default: a label given after seeing the model measures
        <i>agreement</i>, not accuracy. Each label records which it was.</p>
    </div>
  </div>
</main>
<div id=toast></div>
<script>
let Q=null, i=0, cur=null, sel=null, unsure=false, t0=Date.now();
// "always show hint" is PER ROUND and defaults OFF. It used to live under the fixed key
// 'always_v2', so a round enabled in v2 silently stayed enabled in v3 -- every one of the 151 v3
// labels came back assisted=true at a median 2.0s per clip, which turns an accuracy measurement into
// an agreement measurement without anyone choosing that. Keying by VERSION means a new round starts
// blind and the labeller has to opt in again, deliberately.
let shown=false, always=false, hintKey=null;   // set in boot(), once the round version is known
const $=s=>document.querySelector(s);
function toast(m){const t=$('#toast');t.textContent=m;t.style.opacity=1;setTimeout(()=>t.style.opacity=0,900)}
async function boot(){
  Q=await (await fetch('/api/queue')).json();
  hintKey='always_'+Q.version;                 // per-round, so a new round starts blind
  always=localStorage.getItem(hintKey)==='1';
  $('#btns').innerHTML=Q.labels.map((l,n)=>
    `<button data-l="${l.replace(/"/g,'&quot;')}">${l}<kbd>${n+1}</kbd></button>`).join('');
  $('#btns').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(b.dataset.l));
  $('#bno').onclick=()=>choose(null);
  $('#bu').onclick=()=>{unsure=!unsure;$('#bu').classList.toggle('sel',unsure)};
  $('#bprev').onclick=()=>go(i-1); $('#bskip').onclick=()=>save(true);
  $('#bhint').onclick=showHint;
  $('#always').checked=always;
  $('#always').onchange=e=>{always=e.target.checked;
    localStorage.setItem(hintKey,always?'1':'0'); if(always) showHint();};
  // resume at the first unlabelled clip
  i=Q.items.findIndex(x=>!(x.clip in Q.done)); if(i<0) i=0;
  go(i);
}
function go(n){
  if(n<0||n>=Q.items.length) return;
  i=n; cur=Q.items[i]; sel=null; unsure=false; t0=Date.now();
  shown=false; $('#hint').style.display='none'; $('#hint').innerHTML='';
  $('#bu').classList.remove('sel'); $('#note').value='';
  document.querySelectorAll('#btns button,#bno').forEach(b=>b.classList.remove('sel'));
  const prev=Q.done[cur.clip];
  if(prev){ // revisiting: restore, so a correction does not silently become a new blind label
    sel=prev.ethogram; unsure=!!prev.unsure; $('#note').value=prev.note||'';
    $('#bu').classList.toggle('sel',unsure);
    if(prev.present===false) $('#bno').classList.add('sel');
    else if(sel) [...document.querySelectorAll('#btns button')].find(b=>b.dataset.l===sel)?.classList.add('sel');
  }
  $('#v').src='/clip?path='+encodeURIComponent(cur.clip);
  $('#pos').textContent=`${i+1} / ${Q.items.length}`;
  $('#ver').textContent=Q.version||'';
  $('#info').innerHTML=cur.on_disk? `${cur.camera} · ${cur.date} · <span class=meta>${cur.clip}</span>`
                                  : `<span class=warn>file missing on disk</span> · ${cur.clip}`;
  const done=Object.keys(Q.done).length;
  $('#cnt').textContent=`${done} labelled`; $('#fill').style.width=(100*done/Q.items.length)+'%';
  if(always) showHint();
}
async function showHint(){
  if(!cur) return;
  const h=await (await fetch('/api/hint?clip='+encodeURIComponent(cur.clip))).json();
  shown=true;   // recorded on the label as `assisted`
  const conf=h.unanimous? 'unanimous' : (h.low_conf? 'LOW confidence':'split');
  $('#hint').innerHTML=`<b>model:</b> ${h.ethogram||'-'}<br>`+
    `<span class=meta>votes ${h.votes||'-'} · margin ${h.margin??'-'} · ${conf}</span>`;
  $('#hint').style.display='block';
}
function choose(l){
  sel=l;
  document.querySelectorAll('#btns button,#bno').forEach(b=>b.classList.remove('sel'));
  if(l===null) $('#bno').classList.add('sel');
  else [...document.querySelectorAll('#btns button')].find(b=>b.dataset.l===l)?.classList.add('sel');
  save(false);
}
async function save(skip){
  if(!cur) return;
  const body={clip:cur.clip, present: skip?null:(sel!==null), ethogram: skip?null:sel,
              unsure:unsure, note:$('#note').value, seconds:(Date.now()-t0)/1000, skipped:!!skip,
              assisted: shown};
  const r=await (await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body)})).json();
  Q.done[cur.clip]=body; toast(skip?'skipped':'saved '+r.n_done);
  setTimeout(()=>go(i+1),150);
}
addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA') return;
  if(e.key>='1'&&e.key<='7'){const b=document.querySelectorAll('#btns button')[+e.key-1]; if(b) choose(b.dataset.l);}
  else if(e.key==='0') choose(null);
  else if(e.key==='u') $('#bu').click();
  else if(e.key==='h') showHint();
  else if(e.key==='s') save(true);
  else if(e.key==='ArrowRight') go(i+1);
  else if(e.key==='ArrowLeft') go(i-1);
});
boot();
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    n = len(sample()); done = len(load_out())
    print(f"round {VERSION}: {n} clips | already labelled: {done} -> {OUT.name}")
    print("open http://localhost:8021   (1-7 behaviour · 0 none · u unsure · s skip · arrows nav)")
    uvicorn.run(app, host="0.0.0.0", port=8021, log_level="warning")
