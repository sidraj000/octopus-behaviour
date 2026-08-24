#!/usr/bin/env python3
"""Compute the behaviour analysis from data/behaviour_records.json.
Joins to the index for absolute clock hour. Emits behaviour_stats.json."""
import json, os, collections
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1])
REC=ROOT+"/data/behaviour_records.json"
IDX=ROOT+"/src/octopus_clips_verified.json"
OUT=ROOT+"/data/behaviour_stats.json"

# arousal rubric (transparent, editable): 0..1
ACT_AROUSAL={"still":0.0,"low":0.33,"moderate":0.66,"high":1.0}
POS_AROUSAL={"contracted":0.0,"neutral":0.3,"arms_extended":0.6,"climbing":0.8,"flattened_spread":1.0,"uncertain":0.3}
def arousal(s):
    return round(0.6*ACT_AROUSAL.get(s["activity"],0.33)+0.4*POS_AROUSAL.get(s["posture"],0.3),3)

def main():
    rec=json.load(open(REC))
    idx=json.load(open(IDX)); clips=idx if isinstance(idx,list) else idx.get("clips",idx)
    meta={c["clip_path"]:c for c in clips}
    def abshour(cp):
        c=meta.get(cp)
        if not c: return None
        s=c["segment"]
        try: hh,mm,ss=int(s[0:2]),int(s[2:4]),int(s[4:6])
        except: return None
        return ((hh*3600+mm*60+ss+c.get("start_sec",0))//3600)%24

    present=[(cp,r) for cp,r in rec.items() if r.get("struct",{}).get("present")]
    S={"n_records":len(rec),"n_present":len(present)}

    # 1. activity budget
    S["activity_budget"]=dict(collections.Counter(r["struct"]["behavior"] for _,r in present).most_common())
    # 2. posture / activity / location distributions
    for f in ["posture","activity","location","context"]:
        S[f+"_dist"]=dict(collections.Counter(r["struct"][f] for _,r in present).most_common())
    # 3. circadian: present rate + mean arousal + behavior mix by hour
    #    denominator = ALL extracted windows that hour (from index)
    allh=collections.Counter(); presh=collections.Counter(); aroh=collections.defaultdict(list)
    behh=collections.defaultdict(collections.Counter)
    for c in clips:
        h=abshour(c["clip_path"])
        if h is not None: allh[h]+=1
    for cp,r in present:
        h=abshour(cp)
        if h is None: continue
        presh[h]+=1; aroh[h].append(arousal(r["struct"])); behh[h][r["struct"]["behavior"]]+=1
    S["circadian"]={str(h):{"all":allh[h],"present":presh[h],
                    "present_rate":round(presh[h]/allh[h],3) if allh[h] else 0,
                    "mean_arousal":round(sum(aroh[h])/len(aroh[h]),3) if aroh[h] else None}
                    for h in range(24) if allh[h]>=5}
    # 4. stimulus response: metrics by context
    byctx={}
    for ctx in ["none","human_present","enrichment_object","feeding"]:
        sub=[r for _,r in present if r["struct"]["context"]==ctx]
        if not sub: continue
        byctx[ctx]={"n":len(sub),
            "mean_motion":round(sum(r["mean_motion"] for r in sub)/len(sub),4),
            "mean_arousal":round(sum(arousal(r["struct"]) for r in sub)/len(sub),3),
            "top_behaviors":dict(collections.Counter(r["struct"]["behavior"] for r in sub).most_common(3)),
            "top_postures":dict(collections.Counter(r["struct"]["posture"] for r in sub).most_common(3))}
    S["stimulus_response"]=byctx
    # 5. colour state on colour cameras
    cc=[r for _,r in present if not r["grey"]]
    S["colour"]={"n_colour_clips":len(cc),
        "body_color_dist":dict(collections.Counter(r["struct"]["body_color"] for r in cc).most_common()),
        "change_dist":dict(collections.Counter(r["struct"]["color_or_texture_change"] for r in cc).most_common()),
        # does colour vary by context? (arousal/threat proxy)
        "color_by_context":{ctx:dict(collections.Counter(r["struct"]["body_color"] for r in cc if r["struct"]["context"]==ctx).most_common())
                            for ctx in ["none","human_present","enrichment_object"]}}
    # 6. per-camera activity budget
    S["by_camera"]={cam:dict(collections.Counter(r["struct"]["behavior"] for _,r in present if r["camera"]==cam).most_common())
                    for cam in sorted(set(r["camera"] for _,r in present))}
    json.dump(S, open(OUT,"w"), indent=1)
    print(f"present={S['n_present']}/{S['n_records']}")
    print("activity budget:", S["activity_budget"])
    print("stimulus response:")
    for ctx,d in byctx.items(): print(f"  {ctx:16s} n={d['n']:4d} motion={d['mean_motion']} arousal={d['mean_arousal']} beh={list(d['top_behaviors'])[:2]}")
    print("colour by context:", S["colour"]["color_by_context"])
    print("circadian (hr: rate, arousal):", {h:(d["present_rate"],d["mean_arousal"]) for h,d in list(S["circadian"].items())})
    print(f"wrote {OUT}")

if __name__=="__main__": main()
