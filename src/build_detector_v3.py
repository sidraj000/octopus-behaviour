"""build_detector_v3.py — retrain the presence gate on VLM-labelled clips, without forgetting.

WHY. The presence probe is the FIRST gate in the cascade and the dirtiest: 42.7% of the clips it
passes contain no animal. It was trained on 11,224 frames from 29 source videos over 8 dates. The
5-pass 235B ensemble has since labelled 5,222 clips, of which 1,441 are UNANIMOUSLY absent (all five
passes agree) across 174 source videos -- roughly 6x the video diversity, labelled by a model that
demonstrably beats this probe at presence (it rejects 63% of clips the probe called verified).

WHAT THIS CAN AND CANNOT FIX. Every ensemble clip exists because the OLD probe fired on it, so the
new negatives live inside the old probe's operating region. It can therefore reduce FALSE POSITIVES;
it structurally cannot reduce false negatives, because there are no examples of frames the old probe
wrongly rejected. Claims must be worded accordingly.

--------------------------------------------------------------------------------------------------
THREE DESIGN DECISIONS, each forced by something this project already got wrong once.

1. ANCHOR AGAINST FORGETTING, and MEASURE it. Adding ~5,000 new negatives to a model whose training
   set was 35% positive would drag it toward "absent". The fix is not class weights -- this project
   measured three separate times that rebalancing hurts the class it is meant to protect -- but to
   keep the ORIGINAL training distribution in the mix as an anchor, and then to EVALUATE on held-out
   ORIGINAL positives. A negatives-only test set cannot detect forgetting, which is the exact failure
   being guarded against.

   The anchor widens temporal coverage from the ensemble's single week to eight dates across six
   months (2025-10 to 2026-04), so it is not redundant data. But it is NOT disjoint: 3 sessions
   appear in both corpora, which is why both are split in ONE pass over a shared video-key namespace
   (below) rather than independently. Splitting them separately -- the obvious implementation -- would
   have put an anchor frame from an ensemble test video into training.

2. TEST ON VIDEOS THE MODEL HAS NEVER SEEN. Splits are by SOURCE VIDEO on both corpora -- (date,
   session, camera) for the original frames, (date, segment) for the clips. A held-out frame from a
   training video is not held out: same tank, same lighting, minutes apart.

3. LETTERBOX, NOT CENTRE CROP. CLIP's default CenterCrop drops 33-44% of a 16:9 frame and was the
   original root cause of poor field performance here. Features are computed on letterboxed frames.
   The cached `clip_features*.npz` are keyed on PATH, not on the transform, so they are NOT reused --
   reusing them would silently mix two preprocessings.

Frame-level label noise is asymmetric and handled honestly: an ABSENT clip is absent in every frame,
so all new negatives are clean; a PRESENT clip may contain frames where the animal is hidden, so the
new positives carry some noise. We do NOT filter those with the old probe, because that would train
the new model only on positives the old one already gets right, reinforcing its blind spots.

Usage:
  venv/bin/python3 src/build_detector_v3.py --stage features     # extract + embed (slow, resumable)
  venv/bin/python3 src/build_detector_v3.py --stage train        # split, train, evaluate
"""
import argparse, collections, csv, json, math, os, random, re, sys, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import caption_openrouter as C
from ensemble_235b import extract_frames_at, DENSE_FPS
from build_ethogram_dataset import ROOTS

VOTED = REPO / "data" / "ensemble_235b_voted.json"
ORIG = REPO / "data" / "octopus-classification" / "frames"
OUT = REPO / "data" / "detector_v3"
N_FRAMES = 5                 # per clip, at equal intervals, as specified
TARGET_TRAIN_ABSENT = 1000   # target clip count in train; filled by whole videos so it is approximate
SEED = 20260823


# ----------------------------------------------------------------- sources
# The anchor corpus carries THREE filename conventions, and a single-pattern parser silently
# discarded 2,566 of 7,952 present frames (a third of the anchor set):
#   1  2026-04-05_2015_Right_Front_0104-0112_0001.jpg      clip-extracted frames
#   2  p0.50_2026-04-07_190003_Right_Top_t0830.jpg          scanner frames (score-prefixed)
#   3  hardneg_p0.87_2026-02-21_123002_Right_Back_...f01.jpg  mined hard negatives
# All three encode date, session and camera, which is all the split needs. Parsing only the first
# also HID A LEAK: patterns 2-3 include 2026-02-20 sessions that also appear in the ensemble corpus,
# so the video key is normalised to `date/session` -- the SAME key space the clips use -- and any
# session present in both corpora is forced onto one side of the split.
_PATS = [
    re.compile(r"^(?:hardneg_)?(?:p[\d.]+_)?(\d{4}-\d{2}-\d{2})_(\d+)_([A-Za-z]+_[A-Za-z]+)"),
]


def parse_src(name):
    for pat in _PATS:
        m = pat.search(name)
        if m:
            return m.group(1), m.group(2), m.group(3)
    return None


def orig_rows():
    """Original detector frames that are actually on disk, with a parsed source video."""
    rows, missing, unparsed = [], 0, 0
    for r in csv.DictReader(open(ORIG / "manifest.csv")):
        rel = Path(r["path"])
        try:
            f = ORIG / rel.relative_to("data/frames")
        except ValueError:
            f = ORIG / rel.name
        if not f.exists():
            missing += 1; continue
        m = parse_src(f.name)
        if not m:
            unparsed += 1; continue
        date, sess, _cam = m
        # SAME key space as the ensemble clips, so a session in both corpora cannot straddle the split
        rows.append({"path": str(f), "label": 1 if r["label"] == "visible" else 0,
                     "video": f"{date}/{sess}", "src": "orig"})
    print(f"original frames: {len(rows)} usable "
          f"({missing} manifest entries absent from disk, {unparsed} unparseable names), "
          f"{len({r['video'] for r in rows})} source videos")
    return rows


def clip_rows():
    """Unanimous 5/5 ensemble clips -> per-clip records. Absent are clean; present carry frame noise."""
    v = json.load(open(VOTED))

    def votes(k):
        a, b = (v[k].get("present_votes") or "0/0").split("/")
        return int(a), int(b)

    out = []
    for k, x in v.items():
        if votes(k) != (5, 5):
            continue
        if x.get("present") is False:
            out.append({"clip": k, "label": 0, "video": "/".join(k.split("/")[:2])})
        elif x.get("present") is True:
            out.append({"clip": k, "label": 1, "video": "/".join(k.split("/")[:2])})
    a = sum(1 for r in out if r["label"] == 0)
    print(f"ensemble 5/5 clips: {a} absent + {len(out)-a} present, "
          f"{len({r['video'] for r in out})} source videos")
    return out


def resolve(clip):
    for r in ROOTS:
        p = r / clip
        if p.exists() and p.stat().st_size > 10000:
            return p
    return None


# ----------------------------------------------------------------- split
def split_by_video(clips, rng):
    """Whole videos to train/test, filling train until TARGET_TRAIN_ABSENT absent clips are reached.

    Videos are the unit, so no video contributes to both sides. Absent-clip count drives the fill
    because negatives are the scarce, clean signal this retrain exists to add.
    """
    byv = collections.defaultdict(list)
    for r in clips:
        byv[r["video"]].append(r)
    vids = sorted(byv)
    rng.shuffle(vids)
    train, test, n_abs = set(), set(), 0
    for vd in vids:
        if n_abs < TARGET_TRAIN_ABSENT:
            train.add(vd); n_abs += sum(1 for r in byv[vd] if r["label"] == 0)
        else:
            test.add(vd)
    return train, test


# ----------------------------------------------------------------- features
def load_clip_model():
    cm, pre, _clf, _vis, dev = C.load_detector()
    return cm, pre, dev


@torch.no_grad()
def embed(paths, cm, pre, dev, bs=64):
    outs = []
    for i in range(0, len(paths), bs):
        batch = [pre(C.letterbox(Image.open(p).convert("RGB"))) for p in paths[i:i + bs]]
        f = cm.encode_image(torch.stack(batch).to(dev)).float()
        outs.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
    return np.concatenate(outs) if outs else np.zeros((0, 512), np.float32)


def stage_features(a):
    OUT.mkdir(parents=True, exist_ok=True)
    cm, pre, dev = load_clip_model()
    print(f"CLIP on {dev}; letterbox preprocessing (cached npz deliberately NOT reused)")

    # --- original frames ---
    fo = OUT / "orig_feats.npz"
    if not fo.exists():
        rows = orig_rows()
        X = embed([r["path"] for r in rows], cm, pre, dev)
        np.savez_compressed(fo, X=X, y=np.array([r["label"] for r in rows]),
                            video=np.array([r["video"] for r in rows]))
        print(f"wrote {fo}  {X.shape}")
    else:
        print(f"{fo.name} exists, skipping")

    # --- clip frames: 5 at equal intervals, resumable per clip ---
    fc = OUT / "clip_feats"
    fc.mkdir(exist_ok=True)
    rows = clip_rows()
    if a.limit:
        rows = rows[:a.limit]
    done = {p.stem for p in fc.glob("*.npy")}
    todo = [r for r in rows if r["clip"].replace("/", "__") not in done]
    print(f"clips needing frames: {len(todo)} of {len(rows)}")
    ok = fail = 0
    for n, r in enumerate(todo, 1):
        src = resolve(r["clip"])
        if src is None:
            fail += 1; continue
        try:
            with tempfile.TemporaryDirectory() as td:
                fr = extract_frames_at(src, td, DENSE_FPS)
                if len(fr) < N_FRAMES:
                    fail += 1; continue
                idx = np.linspace(0, len(fr) - 1, N_FRAMES).round().astype(int)   # equal intervals
                X = embed([fr[i] for i in idx], cm, pre, dev)
                np.save(fc / (r["clip"].replace("/", "__") + ".npy"), X)
                ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  FAIL {r['clip']}: {type(e).__name__}: {e}")
        if n % 200 == 0:
            print(f"  {n}/{len(todo)}  ok={ok} fail={fail}", flush=True)
    print(f"clip features: {ok} new, {fail} failed, {len(list(fc.glob('*.npy')))} total")


# ----------------------------------------------------------------- train
class Probe(nn.Module):
    """mlp_256_64 -- identical to clip_mlp_hardneg_v2 so the comparison is architecture-free."""

    def __init__(self, d=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def metrics(prob, y):
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(1, tp + fn); fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    order = np.argsort(prob); ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)) if n1 and n0 else float("nan")
    return {"recall": rec, "fpr": fpr, "precision": prec, "auc": auc,
            "n_pos": int(n1), "n_neg": int(n0)}


def stage_train(a):
    rng = random.Random(SEED)
    o = np.load(OUT / "orig_feats.npz", allow_pickle=True)
    Xo, yo, vo = o["X"], o["y"], o["video"]
    clips = clip_rows()
    fc = OUT / "clip_feats"
    clips = [r for r in clips if (fc / (r["clip"].replace("/", "__") + ".npy")).exists()]
    print(f"clips with features: {len(clips)}")

    # ONE split over the UNION of video keys. 3 sessions exist in both corpora, so splitting the two
    # independently would place an anchor frame from an ensemble TEST video into training -- a leak
    # that a per-corpus split cannot see because each side looks clean on its own.
    tr_v, te_v = split_by_video(clips, rng)
    anchor_vids = set(vo.tolist())
    unseen = sorted(anchor_vids - tr_v - te_v)          # anchor-only sessions
    rng.shuffle(unseen)
    n_tr = int(round(0.75 * len(unseen)))
    otr = tr_v | set(unseen[:n_tr])                     # overlapping sessions inherit the clip split
    ote = te_v | set(unseen[n_tr:])
    assert not (otr & ote), "video appears in both splits"
    shared = anchor_vids & (tr_v | te_v)
    print(f"ensemble videos: {len(tr_v)} train / {len(te_v)} test")
    print(f"anchor videos  : {len(anchor_vids)} total, {len(shared)} shared with the clip corpus "
          f"(forced onto the clip side), {len(unseen)} anchor-only")

    def gather(clip_vids, orig_vids):
        Xs, ys, tag = [], [], []
        m = np.isin(vo, list(orig_vids))
        Xs.append(Xo[m]); ys.append(yo[m]); tag += ["orig"] * int(m.sum())
        for r in clips:
            if r["video"] in clip_vids:
                X = np.load(fc / (r["clip"].replace("/", "__") + ".npy"))
                Xs.append(X); ys.append(np.full(len(X), r["label"])); tag += ["ens"] * len(X)
        return np.concatenate(Xs), np.concatenate(ys), np.array(tag)

    Xtr, ytr, ttr = gather(tr_v, otr)
    Xte, yte, tte = gather(te_v, ote)
    for nm, X, y, t in (("train", Xtr, ytr, ttr), ("test", Xte, yte, tte)):
        print(f"{nm}: {len(y)} frames  pos {int(y.sum())} / neg {int((y==0).sum())}"
              f"  | anchor {int((t=='orig').sum())} new {int((t=='ens').sum())}")

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    best, best_state, bad = -1.0, None, 0
    for ep in range(80):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = lossf(model(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(Xte)), -1)[:, 1].numpy()
        m = metrics(p, yte)
        score = m["auc"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pnew = torch.softmax(model(torch.from_numpy(Xte)), -1)[:, 1].numpy()

    # --- the incumbent, on the SAME frames ---
    ck = torch.load(REPO / "weights" / "clip_mlp_hardneg_v2.pt", map_location="cpu")
    old = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.0),
                        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.0), nn.Linear(64, 2))
    old.load_state_dict(ck["state_dict"]); old.eval()
    with torch.no_grad():
        pold = torch.softmax(old(torch.from_numpy(Xte)), -1)[:, 1].numpy()

    print("\n" + "=" * 74)
    print("TEST = frames from videos NEITHER model was trained on")
    res = {}
    for nm, subset in (("all test frames", np.ones(len(yte), bool)),
                       ("new (ensemble) frames", tte == "ens"),
                       ("ANCHOR (original) frames -- the forgetting check", tte == "orig")):
        if subset.sum() == 0:
            continue
        mo, mn = metrics(pold[subset], yte[subset]), metrics(pnew[subset], yte[subset])
        print(f"\n{nm}  (n={int(subset.sum())}, pos {mo['n_pos']} / neg {mo['n_neg']})")
        print(f"  {'':<10}{'recall':>9}{'FPR':>9}{'precision':>11}{'AUC':>9}")
        print(f"  {'v2 (old)':<10}{mo['recall']:>9.3f}{mo['fpr']:>9.3f}{mo['precision']:>11.3f}{mo['auc']:>9.3f}")
        print(f"  {'v3 (new)':<10}{mn['recall']:>9.3f}{mn['fpr']:>9.3f}{mn['precision']:>11.3f}{mn['auc']:>9.3f}")
        print(f"  {'delta':<10}{mn['recall']-mo['recall']:>+9.3f}{mn['fpr']-mo['fpr']:>+9.3f}"
              f"{mn['precision']-mo['precision']:>+11.3f}{mn['auc']-mo['auc']:>+9.3f}")
        res[nm] = {"old": mo, "new": mn}
    torch.save({"state_dict": model.state_dict(), "feat_dim": 512, "clip_model": "ViT-B/32",
                "arch": "mlp_256_64", "label_map": {"visible": 1, "hidden": 0},
                "trained_on": "original anchor + 5/5-unanimous ensemble clips, letterbox",
                "results": {k: {kk: {m: float(x) for m, x in vv.items()} for kk, vv in v.items()}
                            for k, v in res.items()}},
               REPO / "weights" / "clip_mlp_v3.pt")
    (OUT / "results.json").write_text(json.dumps(
        {k: {kk: {m: float(x) for m, x in vv.items()} for kk, vv in v.items()} for k, v in res.items()},
        indent=1))
    print(f"\nwrote weights/clip_mlp_v3.pt and {OUT/'results.json'}")
    print("\nREAD BEFORE CLAIMING: every ensemble clip was SELECTED by the old probe, so this can")
    print("reduce FALSE POSITIVES but structurally cannot reduce false negatives. If recall on the")
    print("ANCHOR rows dropped, the model forgot and the result must not be reported as an upgrade.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "train"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    (stage_features if a.stage == "features" else stage_train)(a)


if __name__ == "__main__":
    main()
