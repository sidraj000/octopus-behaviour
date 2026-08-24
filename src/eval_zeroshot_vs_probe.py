"""eval_zeroshot_vs_probe.py — what does TRAINING the probe actually buy us?

The paper reports the detector at 96.8% and the presence gate at AUC 0.907, but it has never
compared the trained probe against **untrained CLIP**, so it cannot say how much of the performance
comes from the probe rather than from CLIP's features. The 96.8% headline can't answer it: that test
set was self-selected and has no zero-shot arm.

THE SET (this is the point): EMPTY-V2 — 23 human-confirmed `octopus_present` + 97 human-confirmed
`empty` frames, 60 source videos, sampled at **uniform random timestamps over whole source videos**.
It is therefore **detector-independent**: unlike the 232 mined hard negatives (selected at
p_visible >= 0.70, so their p_visible spans only 0.81-1.0 and any detector AUC on them is a
selection artifact), nothing here was chosen by the model being evaluated.

ARMS, all scored on those identical 120 frames:
  1. trained probe   — clip_mlp_hardneg_v2, letterbox, P(visible). The SHIPPED gate.
  2. zero-shot CLIP  — same frozen CLIP ViT-B/32 backbone, no probe: prompt-ensemble
                       softmax over octopus-vs-empty text prompts.
  3. mask area       — the segmenter's presence signal, for context.
Arms 1 and 2 share the SAME backbone and the same letterbox preprocessing, so the difference
isolates the probe rather than confounding it with a different feature extractor.

Metrics: AUC + FP at fixed present-recall (0.90/0.80) + FP at each arm's DEPLOYED threshold.
FP@recall is the headline per BENCHMARKS.md; AUC is secondary. CIs are cluster-bootstrapped
**by source video for both arms** (R9's correction: grouping only the negatives understates
clustering and flatters the result).

Usage: venv/bin/python3 src/eval_zeroshot_vs_probe.py
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from harvest_stream import letterbox, load_detector
from segment_octopus import OctoSegmenter

NEG = REPO / "data" / "empty_negatives"
DET_CKPT = REPO / "weights" / "clip_mlp_hardneg_v2.pt"
SEG_CKPT = REPO / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"
OUT = REPO / "data" / "zeroshot_vs_probe.json"

# Zero-shot prompt ensembles. Deliberately generous to the BASELINE (an ensemble beats a single
# prompt), so the probe's margin is not an artefact of a strawman prompt. Recorded for reproduction.
P_OCTO = ["a photo of an octopus", "an octopus in an aquarium tank",
          "an octopus on the glass of a tank", "an octopus resting on rocks underwater",
          "a cephalopod with visible arms"]
P_EMPTY = ["an empty aquarium tank", "an empty tank with no animal in it",
           "rocks and sand in an empty aquarium", "an underwater tank wall with no animal",
           "empty water with no creature"]

DET_GATE = 0.60      # shipped extraction threshold on p_visible
AREA_GATE = 0.01     # shipped segmenter presence gate


def auc(pos, neg):
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    o = np.argsort(sc); rk = np.empty(len(sc)); rk[o] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fp_at_recall(pos, neg, r):
    """Threshold set so >= r of positives fire; return the fraction of negatives that also fire."""
    thr = np.quantile(np.asarray(pos, float), 1.0 - r)
    return float((np.asarray(neg, float) >= thr).mean()), float(thr)


def boot_by_video(pos, neg, pv, nv, n=5000, seed=0):
    """Cluster bootstrap over source videos, resampling BOTH arms' clusters."""
    rng = np.random.default_rng(seed)
    vids = np.unique(np.concatenate([pv, nv]))
    out = []
    pos, neg, pv, nv = map(np.asarray, (pos, neg, pv, nv))
    for _ in range(n):
        pick = rng.choice(vids, size=len(vids), replace=True)
        p = np.concatenate([pos[pv == v] for v in pick]) if len(pick) else pos
        q = np.concatenate([neg[nv == v] for v in pick]) if len(pick) else neg
        if len(p) and len(q):
            out.append(auc(p, q))
    lo, hi = np.percentile(out, [2.5, 97.5]) if out else (float("nan"),) * 2
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    rows = [r for r in json.load(open(NEG / "index.json"))["rows"]
            if r.get("human") in ("empty", "octopus_present")]
    rows.sort(key=lambda r: r["key"])
    print(f"EMPTY-V2: {len(rows)} human-labelled frames / "
          f"{len({r['video'] for r in rows})} source videos "
          f"({sum(r['human']=='octopus_present' for r in rows)} present, "
          f"{sum(r['human']=='empty' for r in rows)} empty)")

    M = load_detector(str(DET_CKPT))
    S = OctoSegmenter(str(SEG_CKPT))
    cm, pre, dev = M["cm"], M["pre"], M["dev"]

    # zero-shot text side: encode once
    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib
    with torch.no_grad():
        tok = clip_lib.tokenize(P_OCTO + P_EMPTY).to(dev)
        tf = cm.encode_text(tok).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    n_o = len(P_OCTO)

    res = {k: [] for k in ("probe", "zeroshot", "area")}
    labs, vids = [], []
    for i, r in enumerate(rows):
        p = NEG / r["image"]
        im = Image.open(p).convert("RGB")
        bgr = cv2.imread(str(p))
        x = pre(letterbox(im)).unsqueeze(0).to(dev)
        with torch.no_grad():
            f = cm.encode_image(x).float(); f = f / f.norm(dim=-1, keepdim=True)
            res["probe"].append(float(torch.softmax(M["clf"](f), 1)[0, M["vis"]]))
            # zero-shot: mean similarity per class group, then softmax over the two groups
            sim = (100.0 * f @ tf.T).squeeze(0)
            g = torch.stack([sim[:n_o].mean(), sim[n_o:].mean()])
            res["zeroshot"].append(float(torch.softmax(g, 0)[0]))
        _, area = S.segment(bgr)
        res["area"].append(float(area))
        labs.append(1 if r["human"] == "octopus_present" else 0)
        vids.append(r["video"])
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    labs = np.array(labs); vids = np.array(vids)
    out = {"_meta": {"set": "EMPTY-V2 (human-verified, detector-INDEPENDENT: uniform-random "
                            "timestamps over whole source videos)",
                     "n_pos": int(labs.sum()), "n_neg": int((labs == 0).sum()),
                     "n_videos": int(len(set(vids))),
                     "detector": DET_CKPT.name, "segmenter": SEG_CKPT.name,
                     "zeroshot_prompts": {"octopus": P_OCTO, "empty": P_EMPTY},
                     "note": "probe and zeroshot share the SAME frozen CLIP ViT-B/32 backbone and "
                             "letterbox preprocessing, so the delta isolates the probe. Frame-level; "
                             "the deployed gate acts on 20s windows, so this is a per-frame proxy."},
            "arms": {}}
    for k in ("probe", "zeroshot", "area"):
        s = np.array(res[k])
        pos, neg = s[labs == 1], s[labs == 0]
        pv, nv = vids[labs == 1], vids[labs == 0]
        A = auc(pos, neg); lo, hi = boot_by_video(pos, neg, pv, nv)
        f90, t90 = fp_at_recall(pos, neg, 0.90)
        f80, t80 = fp_at_recall(pos, neg, 0.80)
        d = {"auc": round(A, 4), "ci95": [round(lo, 4), round(hi, 4)],
             "fp_at_recall90": round(f90, 4), "thr_at_recall90": round(t90, 4),
             "fp_at_recall80": round(f80, 4),
             "median_pos": round(float(np.median(pos)), 4),
             "median_neg": round(float(np.median(neg)), 4)}
        if k == "probe":
            d["fp_at_deployed_0.60"] = round(float((neg >= DET_GATE).mean()), 4)
            d["recall_at_deployed_0.60"] = round(float((pos >= DET_GATE).mean()), 4)
        if k == "area":
            d["fp_at_deployed_0.01"] = round(float((neg >= AREA_GATE).mean()), 4)
            d["recall_at_deployed_0.01"] = round(float((pos >= AREA_GATE).mean()), 4)
        out["arms"][k] = d
        print(f"\n{k:>9}: AUC {A:.4f} [{lo:.4f},{hi:.4f}]  FP@R.90 {f90:.3f}  FP@R.80 {f80:.3f}"
              f"  med pos/neg {np.median(pos):.4f}/{np.median(neg):.4f}")
        for extra in ("fp_at_deployed_0.60", "recall_at_deployed_0.60",
                      "fp_at_deployed_0.01", "recall_at_deployed_0.01"):
            if extra in d:
                print(f"           {extra} = {d[extra]}")

    # paired delta probe - zeroshot, same frames, clustered by video
    sp, sz = np.array(res["probe"]), np.array(res["zeroshot"])
    rng = np.random.default_rng(0); uv = np.unique(vids); boot = []
    for _ in range(5000):
        pick = rng.choice(uv, size=len(uv), replace=True)
        m = np.concatenate([np.where(vids == v)[0] for v in pick])
        l, P, Z = labs[m], sp[m], sz[m]
        if l.sum() and (l == 0).sum():
            boot.append(auc(P[l == 1], P[l == 0]) - auc(Z[l == 1], Z[l == 0]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dA = out["arms"]["probe"]["auc"] - out["arms"]["zeroshot"]["auc"]
    out["delta_probe_minus_zeroshot"] = {"mean": round(float(dA), 4),
                                         "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                         "includes_zero": bool(lo <= 0 <= hi)}
    print(f"\nΔAUC (probe − zero-shot) = {dA:+.4f}  CI95 [{lo:+.4f},{hi:+.4f}]"
          f"  {'INCLUDES 0' if lo <= 0 <= hi else 'excludes 0'}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
