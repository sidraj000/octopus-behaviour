#!/usr/bin/env python3
"""Render data/behaviour_stats.json -> data/behaviour_dashboard.html (self-contained)."""
import json, html
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1])
S=json.load(open(ROOT+"/data/behaviour_stats.json"))
OUT=ROOT+"/data/behaviour_dashboard.html"

def esc(x): return html.escape(str(x))
# body-colour representative tones (the category IS a colour → encode literally); labeled + tabled
TONE={"pale_white":"#e8e2d6","light_brown":"#c39a6b","mottled":"#9a8c78","dark_red_brown":"#6e3b2c","uncertain":"#b8b6ad"}

def hbars(data, unit="", pct_of=None, hue="var(--series-1)"):
    if not data: return "<p class='muted'>no data</p>"
    mx=max(data.values()) or 1
    rows=[]
    for k,v in data.items():
        w=v/mx*100
        val=f"{v/pct_of*100:.0f}%" if pct_of else f"{v}{unit}"
        rows.append(f"""<div class="row"><div class="lab">{esc(k)}</div>
        <div class="track"><div class="fill" style="width:{w:.1f}%;background:{hue}"></div></div>
        <div class="val">{val}</div></div>""")
    return "<div class='bars'>"+"".join(rows)+"</div>"

def line_svg(hours, ys, ylab, hue, ymax=None):
    W,H,pl,pb,pt,pr=560,180,34,26,14,12
    xs=[int(h) for h in hours]
    ymax=ymax or (max(ys) if ys else 1) or 1
    def px(i): return pl+(W-pl-pr)*(xs[i]-min(xs))/max(1,(max(xs)-min(xs)))
    def py(v): return pt+(H-pt-pb)*(1-v/ymax)
    pts=" ".join(f"{px(i):.1f},{py(v):.1f}" for i,v in enumerate(ys))
    dots="".join(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.2" fill="{hue}"><title>{xs[i]:02d}:00 — {v:.2f}</title></circle>' for i,v in enumerate(ys))
    grid="".join(f'<line x1="{pl}" x2="{W-pr}" y1="{py(ymax*f):.1f}" y2="{py(ymax*f):.1f}" class="grid"/><text x="{pl-6}" y="{py(ymax*f)+3:.1f}" class="tick" text-anchor="end">{ymax*f:.1f}</text>' for f in (0,0.5,1))
    xticks="".join(f'<text x="{px(i):.1f}" y="{H-8}" class="tick" text-anchor="middle">{xs[i]:02d}</text>' for i in range(0,len(xs),max(1,len(xs)//8)))
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(ylab)} by hour">
      {grid}<polyline points="{pts}" fill="none" stroke="{hue}" stroke-width="2" stroke-linejoin="round"/>{dots}{xticks}
      <text x="{pl}" y="10" class="tick">{esc(ylab)}  (x = clock hour)</text></svg>'''

def stacked(byctx):
    # colour composition per context (colour cameras) — literal tones
    out=[]
    for ctx,comp in byctx.items():
        tot=sum(comp.values()) or 1
        segs="".join(f'<div class="seg" style="width:{v/tot*100:.1f}%;background:{TONE.get(k,"#999")}" title="{esc(k)}: {v}"></div>' for k,v in comp.items())
        out.append(f'<div class="crow"><div class="clab">{esc(ctx)} <span class="muted">n={tot}</span></div><div class="cbar">{segs}</div></div>')
    leg="".join(f'<span class="lg"><i style="background:{TONE[k]}"></i>{k}</span>' for k in TONE if k!="uncertain")
    return "<div class='legend'>"+leg+"</div>"+"".join(out)

circ=S.get("circadian",{})
hrs=sorted(circ.keys(), key=int)
rate=[circ[h]["present_rate"]*100 for h in hrs]
aro=[circ[h]["mean_arousal"] or 0 for h in hrs]
sr=S.get("stimulus_response",{})
sr_aro={k:v["mean_arousal"] for k,v in sr.items()}
sr_mot={k:v["mean_motion"] for k,v in sr.items()}
budget=S.get("activity_budget",{})
np_=S.get("n_present",0)

CSS="""
*{box-sizing:border-box} body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
--grid:#e1e0d9;--series-1:#2a78d6;--series-2:#008300;--series-3:#eb6834;color-scheme:light;
background:var(--plane);color:var(--text-primary);min-height:100vh;padding:28px 20px 60px}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .viz-root{--surface-1:#1a1a19;--plane:#0d0d0d;
--text-primary:#fff;--text-secondary:#c3c2b7;--grid:#2c2c2a;--series-1:#3987e5;--series-2:#008300;--series-3:#d95926;color-scheme:dark}}
:root[data-theme=dark] .viz-root{--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;
--grid:#2c2c2a;--series-1:#3987e5;--series-2:#008300;--series-3:#d95926;color-scheme:dark}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} .sub{color:var(--text-secondary);margin:0 0 24px;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:26px}
.tile{background:var(--surface-1);border:1px solid rgba(128,128,128,.15);border-radius:10px;padding:16px}
.tile .n{font-size:28px;font-weight:650} .tile .l{color:var(--text-secondary);font-size:13px;margin-top:2px}
.card{background:var(--surface-1);border:1px solid rgba(128,128,128,.15);border-radius:12px;padding:18px 20px;margin-bottom:18px}
.card h2{font-size:15px;margin:0 0 3px} .card .csub{color:var(--text-secondary);font-size:13px;margin:0 0 16px}
.bars .row{display:grid;grid-template-columns:200px 1fr 56px;align-items:center;gap:10px;margin:7px 0}
.lab{font-size:13px;color:var(--text-secondary);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.track{background:var(--grid);border-radius:5px;height:16px;overflow:hidden}
.fill{height:100%;border-radius:5px} .val{font-size:13px;font-variant-numeric:tabular-nums;text-align:right}
.chart{width:100%;height:auto} .grid{stroke:var(--grid);stroke-width:1} .tick{fill:var(--muted);font-size:10px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px} @media(max-width:720px){.grid2{grid-template-columns:1fr}}
.crow{display:grid;grid-template-columns:150px 1fr;align-items:center;gap:10px;margin:8px 0}
.clab{font-size:13px;color:var(--text-secondary)} .cbar{display:flex;height:18px;border-radius:5px;overflow:hidden;gap:2px;background:var(--grid)}
.seg{height:100%} .legend{margin-bottom:12px;font-size:12px;color:var(--text-secondary)}
.lg{margin-right:14px} .lg i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.muted{color:var(--muted)} .note{font-size:12px;color:var(--muted);margin-top:10px;line-height:1.5}
"""

def tiles():
    hp=max(circ.items(), key=lambda kv:kv[1]["present_rate"], default=(None,{"present_rate":0}))
    hn = sum(v["n"] for v in sr.values() if v)
    return f"""<div class="tiles">
    <div class="tile"><div class="n">{np_}</div><div class="l">present clips analyzed</div></div>
    <div class="tile"><div class="n">{int(hp[0]) if hp[0] else '–'}:00</div><div class="l">peak activity hour ({hp[1]['present_rate']*100:.0f}% present)</div></div>
    <div class="tile"><div class="n">{sr.get('human_present',{}).get('mean_arousal','–')}</div><div class="l">arousal when human present</div></div>
    <div class="tile"><div class="n">{sr.get('none',{}).get('mean_arousal','–')}</div><div class="l">arousal at baseline</div></div></div>"""

HTML=f"""<div class="viz-root"><div class="wrap">
<h1>Nity — behavioural profile</h1>
<p class="sub">Structured signals extracted per clip (Qwen3-VL-235B) over {np_} present clips, aggregated. Arousal = 0.6·activity + 0.4·posture-spread (transparent rubric).</p>
{tiles()}
<div class="card"><h2>Activity budget</h2><p class="csub">Share of present clips per behaviour (7-class ethogram).</p>
{hbars(budget, pct_of=sum(budget.values()) or 1)}</div>

<div class="grid2">
<div class="card"><h2>Circadian activity</h2><p class="csub">Present-octopus rate by clock hour — normalized by all extracted windows that hour (a rate, not a count).</p>
{line_svg(hrs,rate,"% present",'var(--series-1)')}</div>
<div class="card"><h2>Arousal by hour</h2><p class="csub">Mean arousal index of present clips, by clock hour.</p>
{line_svg(hrs,aro,"arousal",'var(--series-2)',ymax=1)}</div>
</div>

<div class="card"><h2>Response to stimulus</h2><p class="csub">Mean arousal index by social/enrichment context — does the animal ramp up around people & enrichment?</p>
{hbars({k:round(v,3) for k,v in sr_aro.items()}, hue='var(--series-3)')}
<div class="note">Motion energy per context: {", ".join(f"{k} {v:.3f}" for k,v in sr_mot.items())}. Baseline vs human-present is the key contrast.</div></div>

<div class="card"><h2>Body colour by context</h2><p class="csub">Colour cameras only (Right_Back / Right_Front). Tones drawn literally; IR cameras excluded (colour not visible).</p>
{stacked(S.get('colour',{}).get('color_by_context',{}))}
<div class="note">Colour <em>change</em> is not reliably measured yet (needs animal segmentation); this shows static body-tone composition.</div></div>

<p class="note">Presence gate still ~66% dirty upstream — absolute activity levels will shift once the detector is retrained on the not-present verdicts. Circadian/response contrasts are robust to this. Generated from behaviour_records.json.</p>
</div></div>"""

open(OUT,"w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Nity behavioural profile</title><style>{CSS}</style></head><body>{HTML}</body></html>")
print("wrote", OUT)
