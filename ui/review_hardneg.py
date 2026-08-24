"""
Hard-negative review UI — eyeball the 232 candidate frames and label each
"octopus" (visible) vs "no octopus" (true hard negative).

Frames are sorted by the OWLv2 octopus score (high = detector thinks an octopus
is present). p_visible (the CLIP+MLP model's call) and the OWLv2 score are shown
on each card. Decisions are saved to data/hard_negatives/review_decisions.csv.

Segmentation: none available (SAM2 weights not downloaded); OWLv2 score shown instead.

Usage:  venv/bin/python3 ui/review_hardneg.py   ->  http://localhost:8004
"""
import csv, json, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

PROJECT   = Path(__file__).resolve().parent.parent
HN        = PROJECT / "data" / "hard_negatives"
DETJSON   = HN / "_detector_verify.json"
DECISIONS = HN / "review_decisions.csv"

app = FastAPI()


def load_items():
    det = json.load(open(DETJSON))
    items = []
    for r in det["results"]:                      # already sorted by score desc
        fn = r["frame"]
        pv = fn.split("_p", 1)[1][:4] if "_p" in fn else "?"
        items.append({"frame": fn, "owlv2": r["max_score"], "p_visible": pv})
    return items


def load_decisions():
    d = {}
    if DECISIONS.exists():
        for row in csv.DictReader(open(DECISIONS)):
            d[row["frame"]] = row["label"]
    return d


def save_decisions(d, meta):
    with open(DECISIONS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label", "owlv2", "p_visible", "ts"])
        for k, v in d.items():
            m = meta.get(k, {})
            w.writerow([k, v, m.get("owlv2", ""), m.get("p_visible", ""),
                        datetime.datetime.now().isoformat(timespec="seconds")])


@app.get("/img/{name}")
def img(name: str):
    p = HN / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.post("/mark")
async def mark(req: Request):
    body = await req.json()
    items = {i["frame"]: i for i in load_items()}
    d = load_decisions()
    if body.get("label") == "clear":
        d.pop(body["frame"], None)
    else:
        d[body["frame"]] = body["label"]
    save_decisions(d, items)
    vals = list(d.values())
    return JSONResponse({"ok": True, "octopus": vals.count("octopus"),
                         "hardneg": vals.count("hardneg"), "total": len(items)})


@app.get("/", response_class=HTMLResponse)
def index():
    items = load_items()
    dec = load_decisions()
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Hard-negative review</title><style>
 *{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0d0d0d;color:#eee;
   height:100vh;display:flex;flex-direction:column;overflow:hidden}}
 header{{background:#1c1c1c;padding:8px 16px;border-bottom:1px solid #333;display:flex;
   align-items:center;gap:18px;flex-wrap:wrap}}
 h1{{font-size:15px;margin:0}} .counts span{{margin-right:12px;font-size:13px}}
 .keys{{font-size:12px;color:#aaa;margin-left:auto}} .keys b{{color:#ddd}}
 .stage{{flex:1;position:relative;display:flex;align-items:center;justify-content:center;min-height:0;padding:8px}}
 #img{{max-width:100%;max-height:100%;object-fit:contain;border:4px solid #333;border-radius:6px}}
 #img.octopus{{border-color:#2ecc71}} #img.hardneg{{border-color:#e74c3c}}
 .hud{{position:absolute;top:14px;left:18px;background:#000a;padding:6px 10px;border-radius:6px;font-size:13px}}
 .badge{{display:inline-block;font-size:12px;padding:2px 7px;border-radius:4px;margin-right:6px;background:#333}}
 .pv{{background:#2c3e50}}
 .owlhi{{background:#7b241c}} .owlmid{{background:#7d6608}} .owllo{{background:#1b4f72}}
 .verdict{{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);font-size:22px;font-weight:700}}
 .v-octopus{{color:#2ecc71}} .v-hardneg{{color:#e74c3c}}
 footer{{background:#1c1c1c;border-top:1px solid #333;padding:8px;display:flex;gap:10px;justify-content:center}}
 footer button{{border:0;padding:9px 16px;border-radius:6px;color:#fff;cursor:pointer;font-size:14px}}
 .oct{{background:#27632a}} .neg{{background:#7b241c}} .skip{{background:#34495e}} .nav{{background:#222}}
 footer button:hover{{filter:brightness(1.3)}}
</style></head><body>
<header><h1>Hard-negative review</h1>
<div class="counts"><span>pos <b id=pos>1</b>/<b id=tot>0</b></span>
<span style="color:#2ecc71">🐙 <b id=oct>0</b></span>
<span style="color:#e74c3c">∅ <b id=neg>0</b></span>
<span>left <b id=un>0</b></span></div>
<div class="keys"><b>O</b>/1 octopus · <b>N</b>/0 no-octopus · <b>Space</b>/→ skip · <b>←</b> back · <b>U</b> clear · <b>F</b> full-res</div>
</header>
<div class="stage">
  <img id="img">
  <div class="hud"><span id="owl" class="badge"></span><span id="pv" class="badge pv"></span><span id="fn" style="font-size:11px;color:#bbb"></span></div>
  <div id="verdict" class="verdict"></div>
</div>
<footer>
 <button class="nav" onclick="go(-1)">← Back</button>
 <button class="neg" onclick="mark('hardneg')">∅ No octopus (N)</button>
 <button class="oct" onclick="mark('octopus')">🐙 Octopus (O)</button>
 <button class="skip" onclick="go(1)">Skip →</button>
</footer>
<script>
const items={json.dumps(items)};
let dec={json.dumps(dec)};
let i=0;
const $=id=>document.getElementById(id);
// start at first unreviewed
i=items.findIndex(it=>!dec[it.frame]); if(i<0)i=0;
function counts(){{
 const v=Object.values(dec);
 $('oct').textContent=v.filter(x=>x=='octopus').length;
 $('neg').textContent=v.filter(x=>x=='hardneg').length;
 $('un').textContent=items.length-v.length;
 $('tot').textContent=items.length;
}}
function render(){{
 const it=items[i];
 $('img').src='/img/'+it.frame;
 const o=it.owlv2, cls=o>=0.5?'owlhi':o>=0.3?'owlmid':'owllo';
 $('owl').textContent='OWLv2 '+o.toFixed(2); $('owl').className='badge '+cls;
 $('pv').textContent='p_vis '+it.p_visible;
 $('fn').textContent=it.frame;
 $('pos').textContent=i+1;
 const lab=dec[it.frame];
 $('img').className=lab||'';
 $('verdict').className='verdict'+(lab?' v-'+lab:'');
 $('verdict').textContent=lab=='octopus'?'🐙 OCTOPUS':lab=='hardneg'?'∅ NO OCTOPUS':'';
 if(items[i+1])new Image().src='/img/'+items[i+1].frame; // preload next
 counts();
}}
function go(d){{ i=Math.max(0,Math.min(items.length-1,i+d)); render(); }}
async function send(frame,label){{
 await fetch('/mark',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{frame:frame,label:label}})}});
}}
function mark(label){{
 const f=items[i].frame;
 if(dec[f]==label){{ delete dec[f]; send(f,'clear'); render(); return; }}
 dec[f]=label; send(f,label); render(); setTimeout(()=>go(1),120);
}}
function clearOne(){{ const f=items[i].frame; delete dec[f]; send(f,'clear'); render(); }}
document.addEventListener('keydown',e=>{{
 if(e.key=='o'||e.key=='O'||e.key=='1')mark('octopus');
 else if(e.key=='n'||e.key=='N'||e.key=='0')mark('hardneg');
 else if(e.key==' '||e.key=='ArrowRight'){{e.preventDefault();go(1);}}
 else if(e.key=='ArrowLeft')go(-1);
 else if(e.key=='u'||e.key=='U')clearOne();
 else if(e.key=='f'||e.key=='F')window.open($('img').src);
}});
render();
</script></body></html>"""


if __name__ == "__main__":
    print("Hard-negative review → http://localhost:8004")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="warning")
