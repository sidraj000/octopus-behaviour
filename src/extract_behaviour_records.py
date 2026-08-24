#!/usr/bin/env python3
"""Full structured behavioural extraction over all present clips on disk.
Parallel OpenRouter workers, snap-to-list validation, resumable, cost-tracked.
Writes data/behaviour_records.json (keyed by clip_path) — does NOT touch the index."""
import sys, os, json, glob, tempfile, shutil, collections, time, requests, threading
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, HERE)
import caption_openrouter as C

OUT = ROOT + "/data/behaviour_records.json"
WORKERS = 6
BEHAV = {"Resting / stationary","Exploration / manipulation","Crawling","Swimming / jetting",
         "Reaching out of water","Human / enrichment interaction","Colour change / defensive","uncertain"}
POSTURE = {"contracted","neutral","arms_extended","flattened_spread","climbing","uncertain"}
ACTIVITY = {"still","low","moderate","high"}
LOCATION = {"in_den","den_entrance","open_substrate","on_glass","at_surface","uncertain"}
CONTEXT = {"none","human_present","enrichment_object","feeding"}
COLOR = {"pale_white","light_brown","mottled","dark_red_brown","uncertain"}
CHANGE = {"none","lightening","darkening","skin_texture_change","uncertain"}

def snap(v, allowed, default):
    if not isinstance(v, str): return default
    if v in allowed: return v
    vl = v.lower().strip()
    for a in allowed:
        if a.lower() == vl: return a
    for a in allowed:
        if a != "uncertain" and (a.lower() in vl or vl in a.lower()): return a
    return default

def validate(j):
    return {
        "present": bool(j.get("present", False)),
        "behavior": snap(j.get("behavior"), BEHAV, "uncertain"),
        "posture": snap(j.get("posture"), POSTURE, "uncertain"),
        "activity": snap(j.get("activity"), ACTIVITY, "low"),
        "location": snap(j.get("location"), LOCATION, "uncertain"),
        "context": snap(j.get("context"), CONTEXT, "none"),
        "body_color": snap(j.get("body_color"), COLOR, "uncertain"),
        "color_or_texture_change": snap(j.get("color_or_texture_change"), CHANGE, "none"),
        "confidence": float(j.get("confidence", 0) or 0),
    }

def prompt(colour_ok):
    col = ('  "body_color": one of ["pale_white","light_brown","mottled","dark_red_brown","uncertain"],\n'
           '  "color_or_texture_change": one of ["none","lightening","darkening","skin_texture_change","uncertain"],\n'
           if colour_ok else
           '  "body_color": "uncertain" (this clip is infrared greyscale — colour not visible),\n'
           '  "color_or_texture_change": "uncertain",\n')
    return ("These are the clearest frames, in time order, from one short aquarium clip of Nity, an octopus. "
        "Describe ONLY what is visible. Return ONE JSON object, no prose, keys:\n"
        '  "present": true/false,\n'
        '  "behavior": one of ["Resting / stationary","Exploration / manipulation","Crawling","Swimming / jetting","Reaching out of water","Human / enrichment interaction","Colour change / defensive","uncertain"] (pick the octopus BEHAVIOUR, not its posture),\n'
        '  "posture": one of ["contracted","neutral","arms_extended","flattened_spread","climbing","uncertain"],\n'
        '  "activity": one of ["still","low","moderate","high"],\n'
        '  "location": one of ["in_den","den_entrance","open_substrate","on_glass","at_surface","uncertain"],\n'
        '  "context": one of ["none","human_present","enrichment_object","feeding"],\n'
        + col + '  "confidence": 0.0-1.0.\n'
        "If no octopus visible: present=false, uncertain/none elsewhere. Output ONLY the JSON.")

def call(urls, pr):
    content = [{"type":"image_url","image_url":{"url":u}} for u in urls] + [{"type":"text","text":pr}]
    body = {"model": C.OR_MODEL, "temperature": 0, "max_tokens": C.MAX_TOKENS,
            "usage": {"include": True}, "messages":[{"role":"user","content":content}]}
    h = {"Authorization": f"Bearer {C.API_KEY}", "Content-Type":"application/json", "X-Title":"octo-struct"}
    for a in range(5):
        try:
            r = requests.post(C.OR_URL, headers=h, json=body, timeout=120)
        except Exception:
            time.sleep(2**a); continue
        if r.status_code == 200:
            j = r.json(); return j["choices"][0]["message"]["content"].strip(), (j.get("usage",{}) or {})
        if r.status_code in (429,500,502,503): time.sleep(2**a); continue
        raise RuntimeError(f"{r.status_code}:{r.text[:120]}")
    raise RuntimeError("retries exhausted")

def pjson(t):
    t = t.strip()
    if t.startswith("```"): t = t.split("```")[1].lstrip("json").strip()
    i,j = t.find("{"), t.rfind("}"); return json.loads(t[i:j+1])

def greyscale(p):
    ca=cv2.VideoCapture(p); ca.set(cv2.CAP_PROP_POS_FRAMES,int(ca.get(cv2.CAP_PROP_FRAME_COUNT))//2)
    ok,fr=ca.read(); ca.release()
    if not ok: return True
    b,g,r=fr[:,:,0].astype(int),fr[:,:,1].astype(int),fr[:,:,2].astype(int)
    return bool((abs(r-g).mean()+abs(g-b).mean()+abs(r-b).mean())/3 < 6)

clip_lock = threading.Lock()   # MPS not thread-safe
io_lock = threading.Lock()
state = {"done":0, "cost":0.0, "err":0}

def main():
    idx=json.load(open(HERE+"/octopus_clips_verified.json"))
    clips=idx if isinstance(idx,list) else idx.get("clips",idx)
    roots=[HERE+"/octopus_clips_verified", ROOT+"/data/octopus_clips_verified"]
    disk={}
    for r in roots:
        for f in glob.glob(r+"/**/*.mp4",recursive=True): disk.setdefault("/".join(f.split("/")[-3:]),f)
    def cap(e): return e.get("caption_235b") or e.get("caption") or ""
    def pres(e):
        c=cap(e).lower(); return c and "not present" not in c and "not visible" not in c
    def key(e): return "/".join(e["clip_path"].split("/")[-3:])
    todo=[e for e in clips if pres(e) and key(e) in disk]

    records = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo=[e for e in todo if e["clip_path"] not in records]
    total=len(todo)
    print(f"present-on-disk to process: {total} (already done: {len(records)})", flush=True)
    cm,pre,clf,vis,dev=C.load_detector(); print("detector loaded", flush=True)

    def work(e):
        p=disk[key(e)]; tmp=tempfile.mkdtemp()
        try:
            grey=greyscale(p)
            frames=C.extract_frames(p,tmp)
            if not frames: raise RuntimeError("no frames")
            with clip_lock:
                sc=C.score(frames,cm,pre,clf,vis,dev)
            order=sorted(range(len(frames)),key=lambda k:sc[k],reverse=True)[:C.N_KEEP]; order.sort()
            urls=[C.b64_image(frames[k]) for k in order]
            txt,u=call(urls, prompt(not grey))
            rec={"camera":e["camera"],"date":e["date"],"video_timeline":e.get("video_timeline"),
                 "mean_motion":e.get("mean_motion"),"grey":grey,"struct":validate(pjson(txt)),
                 "old_caption":cap(e),"cost":float(u.get("cost",0) or 0)}
            return e["clip_path"], rec, None
        except Exception as ex:
            return e["clip_path"], None, str(ex)[:120]
        finally:
            shutil.rmtree(tmp,ignore_errors=True)

    t0=time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs=[ex.submit(work,e) for e in todo]
        for fut in as_completed(futs):
            cp, rec, err = fut.result()
            with io_lock:
                if rec:
                    records[cp]=rec; state["done"]+=1; state["cost"]+=rec["cost"]
                else:
                    state["err"]+=1
                n=state["done"]+state["err"]
                if n % 25 == 0 or n==total:
                    json.dump(records, open(OUT,"w"))
                    rate=state["done"]/max(1e-9,(time.time()-t0))
                    eta=(total-n)/max(1e-9,rate)/60
                    print(f"[{n}/{total}] ok={state['done']} err={state['err']} ${state['cost']:.3f} "
                          f"{rate*60:.0f}/min eta={eta:.0f}min", flush=True)
    json.dump(records, open(OUT,"w"))
    print("\n"+"#"*60)
    print(f"DONE ok={state['done']} err={state['err']} in {(time.time()-t0)/60:.1f} min  COST=${state['cost']:.3f}")
    ok=[r for r in records.values() if "struct" in r]
    print("behavior:", dict(collections.Counter(r['struct']['behavior'] for r in ok)))
    print("context :", dict(collections.Counter(r['struct']['context'] for r in ok)))
    cc=[r for r in ok if not r['grey']]
    print(f"body_color (colour clips, n={len(cc)}):", dict(collections.Counter(r['struct']['body_color'] for r in cc)))

if __name__=="__main__": main()
