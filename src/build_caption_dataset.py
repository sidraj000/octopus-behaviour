"""
Build a versioned caption-training snapshot from octopus_clips_verified.json.

Caption-only distillation set for the Qwen3-VL-2B student. Per run it:
  1. selects PRESENT + captioned clips that have a local file
     (caption source priority: caption_235b -> caption; drops "octopus not present"),
  2. dedups near-duplicate clips with CLIP embeddings (within source video), reusing the
     clip_embeddings.npz cache (extends it for new clips),
  3. splits train/val BY SOURCE VIDEO (date/segment/camera) so duplicates never leak,
  4. extracts the best-N CLAHE-enhanced frames per clip — identical to what the 235B teacher
     saw at caption time (dense frames -> score with clip_mlp_hardneg_v2 -> top-N by p_visible),
  5. writes src/dataset/vN/  { frames/, train.jsonl, val.jsonl, snapshot.json }.

"Continue when more clips arrive" = re-run this (bigger snapshot v+1) then retrain from base.

Run:  python3 build_caption_dataset.py                 # auto vN, dedup @ 0.93, 10% val
      python3 build_caption_dataset.py --version v1 --dedup-thresh 0.95 --val-frac 0.1
"""
import argparse, json, hashlib, tempfile, datetime, sys, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# reuse the exact teacher-side helpers so training frames match caption-time frames
from caption_openrouter import load_detector, extract_frames, score, enhance, N_KEEP, IMG_MAXSIDE
from dedup_clips import embed_all, greedy_keep, resolve, CACHE

NOT_PRESENT = "octopus not present"     # canonical reject caption
# reject-style captions the VLM sometimes emits despite a behavior-ish label — must NOT enter a present-only set
_REJECT_CAP = re.compile(
    r"not (visible|present) in (any|these|the (provided |given )?(frame|image))"
    r"|^(the |an )?octopus (is )?(not|is not) (visible|present)"
    r"|no octopus (is )?(visible|present)|remains hidden and not visible", re.I)

INDEX        = HERE / "octopus_clips_verified.json"
DATASET_ROOT = HERE / "dataset"


def clip_id(cp: str) -> str:
    return cp.split("octopus_clips_verified/", 1)[-1].replace("/", "_").replace(".mp4", "")


def pick_caption(x, keys):
    """First non-empty, octopus-present caption from the priority key list."""
    for k in keys:
        v = x.get(k)
        if not v:
            continue
        v = str(v).strip()
        if v.lower().startswith("caption:"):
            v = v[8:].strip()
        low = v.lower()
        if _REJECT_CAP.search(v) or "not present" in low or low in ("uncertain", "unknown", ""):
            continue
        return v
    return None


def next_version() -> str:
    DATASET_ROOT.mkdir(exist_ok=True)
    ns = [int(p.name[1:]) for p in DATASET_ROOT.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    return f"v{(max(ns) + 1) if ns else 1}"


def in_val(x, val_frac) -> bool:
    key = f"{x['date']}/{x['segment']}/{x['camera']}"          # whole source video -> one side of the split
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 1000) < int(val_frac * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="dataset version tag (default: auto vN)")
    ap.add_argument("--dedup-thresh", type=float, default=0.93, help="cosine sim; keep if < thresh vs kept (within video)")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--n-frames", type=int, default=N_KEEP)
    ap.add_argument("--caption-keys", default="caption_235b,caption", help="comma-sep priority list")
    ap.add_argument("--not-present", type=int, default=0,
                    help="also include up to N diverse 'octopus not present' clips (teaches the reject)")
    ap.add_argument("--np-thresh", type=float, default=0.95,
                    help="global CLIP-sim threshold for sampling DIVERSE not-present clips")
    ap.add_argument("--limit", type=int, default=None, help="cap kept clips (debug)")
    args = ap.parse_args()
    cap_keys = [k.strip() for k in args.caption_keys.split(",")]

    index = json.load(open(INDEX)); clips = index["clips"]
    cand = []                                                   # (clip_dict, caption)
    for x in clips:
        if x.get("ethogram_label") == "octopus not present":
            continue
        cap = pick_caption(x, cap_keys)
        if not cap:
            continue
        if not resolve(x["clip_path"]).exists():
            continue
        cand.append((x, cap))
    print(f"present + captioned + local file: {len(cand)}", flush=True)
    if not cand:
        sys.exit("nothing to build (has captioning finished?)")

    cm, pre, clf, vis, dev = load_detector()

    if not args.no_dedup:
        emb = embed_all([x for x, _ in cand], cm, pre)          # reuses/extends clip_embeddings.npz
        xmap  = {x["clip_path"]: x   for x, _ in cand}
        capm  = {x["clip_path"]: c   for x, c in cand}
        groups = defaultdict(list)
        for x, _ in cand:
            if x["clip_path"] in emb:
                groups[(x["date"], x["segment"], x["camera"])].append((x["clip_path"], x.get("start_sec", 0)))
        keep = set()
        for g in groups.values():
            keep.update(greedy_keep(g, emb, args.dedup_thresh))
        cand = [(xmap[cp], capm[cp]) for cp in keep]
        print(f"after within-video dedup @ {args.dedup_thresh}: {len(cand)}", flush=True)

    if args.not_present > 0:                       # add a capped, DIVERSE sample of empty clips (reject signal)
        npres = [x for x in clips if x.get("ethogram_label") == NOT_PRESENT and resolve(x["clip_path"]).exists()]
        emb = {}
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=True)
            emb = {p: v for p, v in zip(z["paths"].tolist(), z["embs"])}
        with_emb = [x for x in npres if x["clip_path"] in emb]
        kept = greedy_keep([(x["clip_path"], x.get("start_sec", 0)) for x in with_emb], emb, args.np_thresh)[:args.not_present]
        npm = {x["clip_path"]: x for x in npres}
        for cp in kept:
            cand.append((npm[cp], NOT_PRESENT))
        print(f"+ {len(kept)} diverse not-present clips (of {len(npres)} available, {len(with_emb)} embedded)", flush=True)

    if args.limit:
        cand = cand[:args.limit]

    ver = args.version or next_version()
    dsdir = DATASET_ROOT / ver
    (dsdir / "frames").mkdir(parents=True, exist_ok=True)

    train, val = [], []
    for i, (x, cap) in enumerate(cand, 1):
        cp = resolve(x["clip_path"])
        with tempfile.TemporaryDirectory() as tmp:
            frames = extract_frames(cp, tmp)
            if not frames:
                continue
            sc = score(frames, cm, pre, clf, vis, dev)
            order = sorted(range(len(frames)), key=lambda k: sc[k], reverse=True)[:args.n_frames]
            best  = [frames[k] for k in sorted(order)]          # keep chronological order
            cid, rels = clip_id(x["clip_path"]), []
            for j, f in enumerate(best):
                im = Image.open(f).convert("RGB")
                im.thumbnail((IMG_MAXSIDE, IMG_MAXSIDE))
                im = enhance(im)                                # CLAHE — same as teacher input
                rel = f"frames/{cid}_f{j:02d}.jpg"
                im.save(dsdir / rel, quality=90)
                rels.append(rel)
        rec = {"clip_path": x["clip_path"], "caption": cap, "frames": rels}
        (val if in_val(x, args.val_frac) else train).append(rec)
        if i % 100 == 0 or i == len(cand):
            print(f"  frames {i}/{len(cand)}", flush=True)

    with open(dsdir / "train.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(dsdir / "val.jsonl", "w") as f:
        for r in val: f.write(json.dumps(r) + "\n")
    snap = {
        "version": ver,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_train": len(train), "n_val": len(val),
        "dedup_thresh": None if args.no_dedup else args.dedup_thresh,
        "val_frac": args.val_frac, "n_frames": args.n_frames,
        "caption_keys": cap_keys,
        "clip_paths": [r["clip_path"] for r in train + val],     # reproducible/auditable
    }
    json.dump(snap, open(dsdir / "snapshot.json", "w"), indent=2)
    print(f"\n{ver}: train={len(train)} val={len(val)} frames-dir={dsdir/'frames'}\n"
          f"zip for Colab:  (cd {dsdir} && zip -qr ../{ver}.zip .)", flush=True)


if __name__ == "__main__":
    main()
