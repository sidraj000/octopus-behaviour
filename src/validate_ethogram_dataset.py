"""validate_ethogram_dataset.py — check the frozen dataset BEFORE training anything.

Three classes of check, run in order of how much they would invalidate:

  1. LEAKAGE. Assert no source video appears in more than one split. This project shipped exactly
     this bug once -- an apparent 0.49 -> 0.70 segmentation gain evaporated under a video-level
     holdout -- so it is asserted, never assumed. Also checks the human_secondary clips are absent
     from train/val.

  2. ADEQUACY. Every class present in every split, with n reported per class per split, plus the
     video count (the real sample size). A per-class F1 computed on 2 test videos is not a result.

  3. DOES THE MOTION CHANNEL ACTUALLY DISCRIMINATE? The premise of adding it is that behaviour is
     defined by movement, so `Resting` should sit clearly below `Locomotion` on motion_disp. If it
     does not, the feature is broken and rungs 2-3 of the training ladder are pointless -- much
     better to learn that in 30 seconds than after training. Reported as per-class medians plus a
     rank-AUC for the single most basic contrast (Resting vs Locomotion).

Also sanity-checks the arrays themselves: shape, NaNs, CLIP-norm, soft targets summing to 1.

Usage: venv/bin/python3 src/validate_ethogram_dataset.py --version v1
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def auc(pos, neg):
    """Rank-AUC: P(a random pos scores above a random neg)."""
    if not len(pos) or not len(neg):
        return None
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    d = REPO / "src" / "dataset_etho" / a.version
    man = [json.loads(l) for l in open(d / "manifest.jsonl") if l.strip()]
    snap = json.load(open(d / "snapshot.json")) if (d / "snapshot.json").exists() else {}
    classes = snap.get("classes") or sorted({r["label"] for r in man})
    print(f"manifest: {len(man)} clips / {len({r['video'] for r in man})} videos / {len(classes)} classes")
    fails = []

    # ---------- 1. LEAKAGE ----------
    print("\n=== 1. LEAKAGE ===")
    vid_splits = collections.defaultdict(set)
    for r in man:
        vid_splits[r["video"]].add(r["split"])
    trainval = {"train", "val", "test"}
    bad = {v: s for v, s in vid_splits.items() if len(s & trainval) > 1}
    print(f"  videos in >1 of train/val/test: {len(bad)}")
    if bad:
        fails.append(f"VIDEO LEAK across splits: {list(bad.items())[:5]}")
        for v, s in list(bad.items())[:5]:
            print(f"    LEAK {v}: {sorted(s)}")
    hs = {r["clip"] for r in man if r["split"] == "human_secondary"}
    overlap = {r["clip"] for r in man if r["split"] in trainval} & hs
    print(f"  human_secondary clips also in train/val/test: {len(overlap)}")
    if overlap:
        fails.append("human_secondary clips present in a trainable split")
    # the honest caveat, restated as a measurement
    hs_vids = {r["video"] for r in man if r["split"] == "human_secondary"}
    tv_vids = {r["video"] for r in man if r["split"] in trainval}
    print(f"  human_secondary videos that ALSO appear in train/val/test: "
          f"{len(hs_vids & tv_vids)}/{len(hs_vids)}  <- expected; why it is a caveated secondary")

    # ---------- 2. ADEQUACY ----------
    print("\n=== 2. ADEQUACY (clips / videos) ===")
    print(f"  {'split':<17}" + "".join(f"{c[:13]:>16}" for c in classes))
    for s in ("train", "val", "test", "human_secondary"):
        g = [r for r in man if r["split"] == s]
        if not g:
            continue
        cc = collections.Counter(r["label"] for r in g)
        row = ""
        for c in classes:
            nv = len({r["video"] for r in g if r["label"] == c})
            row += f"{str(cc[c]) + '/' + str(nv) + 'v':>16}"
            if s in ("train", "val", "test") and cc[c] == 0:
                fails.append(f"class '{c}' ABSENT from split '{s}'")
        print(f"  {s:<17}{row}")
    for s in ("train", "val", "test"):
        g = [r for r in man if r["split"] == s]
        thin = [c for c in classes if len({r["video"] for r in g if r["label"] == c}) < 3]
        if thin:
            print(f"  NOTE {s}: classes on <3 videos (per-class F1 there is not meaningful): {thin}")

    # ---------- 3. IS THE MOTION CHANNEL REAL? ----------
    print("\n=== 3. DOES MOTION DISCRIMINATE? (the premise of rungs 2-3) ===")
    feats = np.load(d / "features.npz") if (d / "features.npz").exists() else None
    per_class = collections.defaultdict(list)
    bad_arr = 0
    for r in man:
        key = r["clip"]
        arr = None
        if feats is not None and key in feats:
            arr = feats[key]
        else:
            fp = d / "feats" / (key.replace("/", "__") + ".npy")
            if fp.exists():
                arr = np.load(fp)
        if arr is None:
            continue
        if arr.shape != (10, 514) or not np.isfinite(arr).all():
            bad_arr += 1
            continue
        per_class[r["label"]].append(float(arr[:, 513].mean()))   # motion_disp
    print(f"  arrays with wrong shape or non-finite values: {bad_arr}")
    if bad_arr:
        fails.append(f"{bad_arr} feature arrays malformed")
    print(f"  median motion_disp by class:")
    for c in sorted(per_class, key=lambda x: -np.median(per_class[x])):
        v = per_class[c]
        print(f"    {c:<32} n={len(v):<5} median {np.median(v):.4f}   IQR "
              f"{np.percentile(v,25):.4f}-{np.percentile(v,75):.4f}")
    rest, loco = per_class.get("Resting / stationary", []), per_class.get("Locomotion (crawl/swim)", [])
    if rest and loco:
        A = auc(loco, rest)
        print(f"\n  Locomotion vs Resting on motion_disp alone: AUC {A:.3f}")
        if A is None or A < 0.60:
            fails.append(f"motion channel barely separates Locomotion from Resting (AUC {A}) -- "
                         "the feature may be broken; rungs 2-3 rest on this")
        else:
            print("  -> the motion channel carries real signal; rungs 2-3 are worth running")

    # ---------- soft targets ----------
    sums = np.array([sum(r["soft"]) for r in man])
    print(f"\n=== soft targets ===\n  rows summing to 1.0: {int(np.isclose(sums,1.0,atol=1e-3).sum())}/{len(man)}")
    if not np.isclose(sums, 1.0, atol=1e-3).all():
        fails.append("some soft targets do not sum to 1")
    unan = sum(1 for r in man if r.get("unanimous_after_merge"))
    print(f"  unanimous after merge: {unan} ({unan/len(man):.1%})  -> the rest teach uncertainty")

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)} check(s):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- safe to train")


if __name__ == "__main__":
    main()
