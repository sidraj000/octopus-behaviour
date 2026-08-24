"""extract_backbone_feats.py — swap the frozen backbone, hold everything else fixed.

WHY. Every rung of the ethogram ladder (R27) is a HEAD on frozen CLIP ViT-B/32 features. The ladder
varied the head and found it barely mattered (rungs 1-3 within ~1 std), and that was read as "the
ceiling is video diversity". That conclusion is narrower than the experiment: the REPRESENTATION was
never varied, so the honest statement is "the ceiling is video diversity, GIVEN frozen CLIP features".

Two hypotheses this script exists to separate:

  (a) APPEARANCE QUALITY. CLIP is trained on image-text pairs, optimising for caption alignment
      rather than fine visual detail. A self-supervised image backbone (DINOv2) may encode the animal
      against dim tank background better. -> `--backbone dinov2`

  (b) TIME. CLIP has no notion of motion at all: an octopus crawling slowly gives near-identical
      embeddings 2 s apart. R27 tried to patch this with two hand-computed changed-pixel channels and
      they bought nothing measurable (+0.006), which was read as "motion does not help". The
      alternative reading is that two scalars are a poor stand-in for a representation that models
      time. A video-native backbone tests that directly. -> `--backbone videomae` / `vjepa2`

WHAT IS HELD FIXED, so the only free variable is the backbone:
  * the same clips (the frozen v1 manifest)
  * the same frames -- the dense ffmpeg extraction at DENSE_FPS is reproduced and the SAME
    `frames_used` indices recorded in the manifest are re-selected, so image backbones see exactly
    the frames CLIP saw
  * the same two motion channels, appended identically, so rung 2/3 definitions are unchanged
  * the same video-level splits and the same seeds at training time

OUTPUT `src/dataset_etho/v1/feats_<backbone>/<clip>.npy` of shape [T, D+2], plus a `meta.json`
recording D so the trainer can slice without a hardcoded 512. T may differ by backbone (video models
emit one token per temporal position); rungs 1-2 pool over T and rung 3 is a GRU, so all three are
T-agnostic and remain architecturally identical across backbones.

Resumable per clip, same as the CLIP build.

Usage:
  venv/bin/python3 src/extract_backbone_feats.py --backbone dinov2
  venv/bin/python3 src/extract_backbone_feats.py --backbone videomae --limit 50
"""
import argparse, json, os, queue, shutil, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import caption_openrouter as C
from ensemble_235b import extract_frames_at, DENSE_FPS
from build_ethogram_dataset import motion_features, ROOTS

BACKBONES = {
    # id, kind, n_frames the model consumes (None = per-frame, use the manifest's picks)
    "dinov2":   ("facebook/dinov2-base",            "image", None),
    "dinov2s":  ("facebook/dinov2-small",           "image", None),
    "videomae": ("MCG-NJU/videomae-base",           "video", 16),
    "vjepa2":   ("facebook/vjepa2-vitl-fpc64-256",  "video", 16),
}


def resolve(clip):
    for r in ROOTS:
        p = r / clip
        if p.exists() and p.stat().st_size > 10000:
            return p
    return None


# ---------------------------------------------------------------------------- mask-guided cropping
# Chosen from the corpus, not guessed. Median longest bbox side is 0.216, p75 0.334, p95 0.642.
# PAD=1.5 gives ~3.1x median linear magnification while leaving margin for an off-centre centroid --
# an extended arm pulls the centroid away from the body, and clipping that arm would destroy the very
# signal `Reaching`/`Exploration` depend on. PAD=1.8 was too generous (2.6x, and 6.9% of frames were
# not cropped at all); PAD=1.0 buys 4.0x but risks cutting the animal.
# MIN=0.20 rather than 0.25 because the 0.25 floor was CAPPING magnification at 4x on exactly the
# small animals that need it most, while still retaining some tank context (surface, den) that
# `Reaching` and `Human interaction` plausibly rely on.
CROP_PAD = 1.5
CROP_MIN = 0.20
_MASK_CACHE = {}


def load_crop_boxes(version):
    """Per-clip [T, 4] boxes from the cached mask features -- NO segmenter re-run needed.

    WHY CROP AT ALL. Every backbone sees the animal at ~48x48 px inside a 224 input (27x27 for the
    smaller quartile): median mask bbox longest side is 0.216 of the frame, so >95% of the input is
    tank. Cropping to the animal gives 4.6x linear magnification at the median, 8.2x on small animals.
    This is the same lever that already worked twice here -- segmentation gained +0.14 IoU from 256->512
    because at 256 the octopus was ~40x40 px, and the detector's letterbox-vs-centre-crop fix was a
    framing change, not a new model.

    The box is built from the CENTROID and the bbox dims, both already stored per frame, so this costs
    nothing extra. A SQUARE crop centred on the centroid with PAD=1.8x the longest side, because the
    centroid is not the bbox centre -- generous padding guarantees the animal stays inside even when
    the mass is off-centre (an extended arm pulls the centroid away from the body). Under-padding
    would silently crop the animal in half, which is worse than not cropping.

    valid=0 frames (IR, or no usable mask) fall back to the FULL frame rather than a guessed box.
    """
    if version in _MASK_CACHE:
        return _MASK_CACHE[version]
    mdir = REPO / "src" / "dataset_etho" / version / "feats_mask"
    boxes = {}
    for f in mdir.glob("*.npy"):
        a = np.load(f)
        cx, cy, bw, bh, valid = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 9]
        side = np.clip(CROP_PAD * np.maximum(bw, bh), CROP_MIN, 1.0)
        b = np.stack([cx, cy, side, valid], 1).astype(np.float32)
        boxes[f.stem] = b
    _MASK_CACHE[version] = boxes
    print(f"crop boxes loaded for {len(boxes)} clips (pad={CROP_PAD}, min_side={CROP_MIN})")
    return boxes


def crop_frame(img, box):
    """img PIL -> cropped PIL. box = (cx, cy, side, valid) in normalised coords."""
    cx, cy, side, valid = box
    if valid < 0.5:
        return img                              # no mask: use the full frame, do not guess
    W, H = img.size
    s = float(side) * max(W, H) / 2.0
    x, y = float(cx) * W, float(cy) * H
    l, t = max(0, x - s), max(0, y - s)
    r, b = min(W, x + s), min(H, y + s)
    if r - l < 16 or b - t < 16:
        return img
    return img.crop((int(l), int(t), int(r), int(b)))


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_backbone(name, dev):
    from transformers import AutoModel
    mid, kind, nfr = BACKBONES[name]
    if kind == "image":
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(mid)
    else:
        try:
            from transformers import AutoVideoProcessor
            proc = AutoVideoProcessor.from_pretrained(mid)
        except Exception:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid)
    n_patched = patch_videomae_qkv_bias(model, mid) if name == "videomae" else 0
    if n_patched:
        print(f"  patched {n_patched} attention bias tensors from the checkpoint's q_bias/v_bias names")
    model = model.to(dev).eval()
    return proc, model, kind, nfr


def patch_videomae_qkv_bias(model, mid):
    """Load the attention QKV biases transformers 5.x silently leaves at ZERO.

    VideoMAE's checkpoint stores `attention.attention.q_bias` / `v_bias` (the original implementation
    has no k_bias -- it is a zero buffer by design). transformers 5.12 expects
    `attention.attention.{query,key,value}.bias`, so 36 of the model's 196 tensors -- three biases per
    layer across 12 layers -- come out FRESHLY INITIALISED. It warns, but only in the generic
    "MISSING ... consider training on your downstream task" form that is easy to read past.

    Left unpatched this would not crash and the features would look plausible, which is exactly the
    failure mode worth guarding: the whole point of the experiment is comparing REPRESENTATIONS, so a
    partially-uninitialised encoder would produce a fake negative result for the video backbone and
    the wrong conclusion about whether time helps. Verified separately that the attention WEIGHTS do
    load correctly (bitwise identical to the checkpoint) -- only the biases were affected.
    """
    import torch
    from transformers.utils import cached_file
    try:
        from safetensors.torch import load_file
        ck = load_file(cached_file(mid, "model.safetensors"))
    except Exception:
        try:
            ck = torch.load(cached_file(mid, "pytorch_model.bin"), map_location="cpu",
                            weights_only=True)
        except Exception:
            return 0
    sd, n = model.state_dict(), 0
    with torch.no_grad():
        for key, tensor in sd.items():
            if not key.endswith((".query.bias", ".value.bias", ".key.bias")):
                continue
            base = key.rsplit(".", 2)[0]                       # ...attention.attention
            which = key.rsplit(".", 2)[1]                      # query | key | value
            if which == "key":
                continue                                       # zero by design in the original impl
            src = ck.get(f"videomae.{base}.{which[0]}_bias")
            if src is not None and src.shape == tensor.shape:
                tensor.copy_(src); n += 1
    return n


@torch.no_grad()
def feats_image(paths, proc, model, dev, boxes=None, letterbox=False, size=224):
    """Per-frame embedding -> [T, D]. CLS token if present, else mean over patch tokens.

    LETTERBOX vs the processor's default CENTRE CROP. dinov2-base's processor resizes the shortest
    edge to 256 then centre-crops 224, which on a 1280x720 frame keeps only 43% of the image and
    throws away 116 px from EACH side. That is precisely the bug this project records as the original
    cause of poor detection performance ("preprocessing is letterbox, not crop ... aspect-ratio
    mismatch, not architecture, was the root cause"), and only the CLIP path had the fix. The octopus
    is frequently at the frame edge on Right_Back/Right_Left, so this was discarding the animal
    outright on some clips.

    With letterbox=True the frame is padded to square at `size` and the processor's resize/crop are
    switched OFF, so nothing is discarded. Note this ALSO reframes the crop result: mask crops are
    square by construction, so they never lost anything to the centre crop -- part of "cropping helps"
    may simply have been "the animal ends up centred". This flag is what separates the two.
    """
    ims = [Image.open(p).convert("RGB") for p in paths]
    if boxes is not None:
        ims = [crop_frame(im, b) for im, b in zip(ims, boxes)]
    if letterbox:
        ims = [C.letterbox(im, size) for im in ims]
        px = proc(images=ims, return_tensors="pt", do_resize=False,
                  do_center_crop=False)["pixel_values"].to(dev)
    else:
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
    h = model(pixel_values=px).last_hidden_state            # [T, tokens, D]
    return h[:, 0].float().cpu().numpy() if h.shape[1] > 1 else h.mean(1).float().cpu().numpy()


@torch.no_grad()
def feats_video(paths, proc, model, dev, n_frames, boxes=None):
    """Video backbone -> [T', D] by pooling spatial tokens at each temporal position.

    Video models emit spatiotemporal tokens; averaging over space keeps a TIME axis so rung 3's GRU
    still has a sequence to model. Collapsing to one vector per clip would quietly turn rung 3 into
    rung 1 and make the comparison meaningless.
    """
    idx = np.linspace(0, len(paths) - 1, n_frames).round().astype(int)
    # numpy frames, and the batch is a LIST of videos -- proc(videos=[frames]) is the only signature
    # transformers 5.x accepts here; PIL input or the pixel_values_videos key both fail.
    _ims = [Image.open(paths[i]).convert("RGB") for i in idx]
    if boxes is not None:
        # video models need a fixed size per clip, so crop then resize every frame to the first
        # frame's crop size -- a per-frame size change would break the tensor stack.
        _ims = [crop_frame(im, boxes[min(k, len(boxes) - 1)]) for k, im in enumerate(_ims)]
        tgt = _ims[0].size
        _ims = [im if im.size == tgt else im.resize(tgt, Image.BILINEAR) for im in _ims]
    vid = [np.array(im) for im in _ims]
    enc = proc(videos=[vid], return_tensors="pt")
    key = "pixel_values_videos" if "pixel_values_videos" in enc else "pixel_values"
    px = enc[key].to(dev)
    out = model(**{key: px}) if key == "pixel_values_videos" else model(pixel_values=px)
    h = out.last_hidden_state[0]                            # [tokens, D]
    d = h.shape[-1]
    # tokens factor as (temporal x spatial); recover the temporal length from the token count
    for t in (n_frames // 2, n_frames, 8, 4):
        if t and h.shape[0] % t == 0:
            return h.reshape(t, -1, d).mean(1).float().cpu().numpy()
    return h.mean(0, keepdim=True).float().cpu().numpy()


def prefetch(rows, q, stop):
    """Producer: ffmpeg-decode clips into temp dirs and queue them for the model thread.

    PROFILED FIRST, then optimised. Per clip: ffmpeg 1.35s (61%), model 0.68s (31%), motion 0.17s
    (8%). The obvious fix -- extract only the 10 needed frames instead of the dense 50 -- gives just
    1.1x, because the cost is DECODING the clip, not writing JPEGs; the fps filter walks the whole
    video either way. (Verified separately that a select-filter extraction is byte-identical to the
    dense one, so that route was correct, just not worth much.)

    What actually helps: decode is CPU-bound and the model runs on MPS, so they should overlap. A
    small thread pool of ffmpeg subprocesses keeps the GPU fed while cores decode. Subprocesses hold
    no GIL, so threads are enough. The queue is BOUNDED -- an unbounded one would let the producers
    run ahead and fill the disk with tens of thousands of JPEGs.
    """
    for r in rows:
        if stop.is_set():
            break
        src = resolve(r["clip"])
        if src is None:
            q.put((r, None, None)); continue
        td = tempfile.mkdtemp(prefix="bbf_")
        try:
            fr = extract_frames_at(src, td, DENSE_FPS)
            q.put((r, fr, td)) if fr else (shutil.rmtree(td, ignore_errors=True), q.put((r, None, None)))
        except Exception:
            shutil.rmtree(td, ignore_errors=True)
            q.put((r, None, None))
    q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    help="one name, or a comma-separated list to run in ONE pass over the frames")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3, help="parallel ffmpeg decoders")
    ap.add_argument("--letterbox", action="store_true",
                    help="pad to square instead of the processor's destructive centre-crop; "
                         "writes feats_<backbone>lb/")
    ap.add_argument("--crop", action="store_true",
                    help="mask-guided crop before embedding; writes feats_<backbone>crop/")
    ap.add_argument("--no-reuse-motion", action="store_true",
                    help="recompute the motion channels instead of copying them from the CLIP build")
    a = ap.parse_args()
    names = [n.strip() for n in a.backbone.split(",") if n.strip()]
    for n in names:
        if n not in BACKBONES:
            sys.exit(f"unknown backbone {n!r}; choose from {sorted(BACKBONES)}")

    d = REPO / "src" / "dataset_etho" / a.version
    man = [json.loads(l) for l in open(d / "manifest.jsonl") if l.strip()]
    if a.limit:
        man = man[:a.limit]
    dev = device()
    # Every requested backbone shares ONE decode of each clip. Decoding is 61% of the per-clip cost,
    # so running two backbones in separate passes pays it twice for nothing.
    B = {}
    for nm in names:
        proc, model, kind, nfr = load_backbone(nm, dev)
        sfx = ("crop" if a.crop else "") + ("lb" if a.letterbox else "")
        o = d / f"feats_{nm}{sfx}"; o.mkdir(parents=True, exist_ok=True)
        B[nm] = {"proc": proc, "model": model, "kind": kind, "nfr": nfr, "out": o,
                 "params": sum(p.numel() for p in model.parameters()),
                 "done": {p.stem for p in o.glob("*.npy")}, "D": None, "ok": 0, "fail": 0}
        print(f"{nm} ({BACKBONES[nm][0]}) {kind} | {B[nm]['params']/1e6:.0f}M params | "
              f"already done {len(B[nm]['done'])}")

    # Motion channels are IDENTICAL by construction to the CLIP build's, so copy them from
    # feats/*.npy columns -2: instead of re-running cv2 over the frames. Not an approximation --
    # the same numbers, and it also removes the last reason to touch every dense frame.
    CROPS = load_crop_boxes(a.version) if a.crop else {}
    globals()["CROPS"] = CROPS
    clip_feats = d / "feats"
    reuse = not a.no_reuse_motion and clip_feats.exists()

    todo = [r for r in man if any(r["clip"].replace("/", "__") not in B[nm]["done"] for nm in names)]
    print(f"clips: {len(man)}   needing work: {len(todo)}   decoders: {a.workers}   "
          f"motion: {'copied from the CLIP build' if reuse else 'recomputed'}")
    if not todo:
        print("nothing to do"); return

    q, stop = queue.Queue(maxsize=a.workers * 2), threading.Event()
    chunks = [todo[i::a.workers] for i in range(a.workers)]
    threads = [threading.Thread(target=prefetch, args=(c, q, stop), daemon=True) for c in chunks]
    for t in threads:
        t.start()

    t0, seen, live = time.time(), 0, len(threads)
    try:
        while live:
            item = q.get()
            if item is None:
                live -= 1; continue
            r, fr, td = item
            seen += 1
            try:
                if not fr:
                    for nm in names:
                        B[nm]["fail"] += 1
                    continue
                pick = [i for i in (r.get("frames_used") or []) if i < len(fr)]
                if not pick:
                    for nm in names:
                        B[nm]["fail"] += 1
                    continue
                mot10 = None
                if reuse:
                    cf = clip_feats / (r["clip"].replace("/", "__") + ".npy")
                    if cf.exists():
                        mot10 = np.load(cf)[:, -2:]
                if mot10 is None or len(mot10) != len(pick):
                    mot10 = motion_features(fr, pick)
                for nm in names:
                    b = B[nm]
                    stem = r["clip"].replace("/", "__")
                    if stem in b["done"]:
                        continue
                    try:
                        bx = None
                        if a.crop:
                            bx = CROPS.get(stem)
                            if bx is None:
                                b["fail"] += 1; continue
                        if b["kind"] == "image":
                            emb = feats_image([fr[i] for i in pick], b["proc"], b["model"], dev, bx,
                                              letterbox=a.letterbox)
                            mot = mot10
                        else:
                            emb = feats_video(fr, b["proc"], b["model"], dev, b["nfr"], bx)
                            tp = np.linspace(0, len(mot10) - 1, emb.shape[0]).round().astype(int)
                            mot = mot10[tp]
                        arr = np.concatenate([emb, mot], axis=1).astype(np.float32)
                        np.save(b["out"] / (stem + ".npy"), arr)
                        b["D"] = arr.shape[-1] - 2; b["ok"] += 1
                    except Exception as e:
                        b["fail"] += 1
                        if b["fail"] <= 3:
                            print(f"  FAIL[{nm}] {r['clip']}: {type(e).__name__}: {e}")
            finally:
                if td:
                    shutil.rmtree(td, ignore_errors=True)
            if seen % 250 == 0:
                rate = seen / max(time.time() - t0, 1e-9) * 60
                eta = (len(todo) - seen) / max(rate, 1e-9)
                print(f"  {seen}/{len(todo)}  {rate:.0f} clips/min  eta {eta:.0f} min  "
                      + " ".join(f"{nm}:ok={B[nm]['ok']},fail={B[nm]['fail']}" for nm in names),
                      flush=True)
    except KeyboardInterrupt:
        stop.set(); print("\ninterrupted -- resumable, rerun to continue")

    for nm in names:
        b = B[nm]
        if b["D"] is None:
            existing = sorted(b["out"].glob("*.npy"))
            if existing:
                b["D"] = int(np.load(existing[0]).shape[-1]) - 2
        if b["D"] is None:
            print(f"  {nm}: no features produced ({b['fail']} failures) -- meta not written")
            continue
        D = b["D"]
        (b["out"] / "meta.json").write_text(json.dumps({
            "backbone": nm, "model_id": BACKBONES[nm][0], "kind": b["kind"], "feat_dim": D,
            "n_motion": 2, "params_millions": round(b["params"] / 1e6, 1),
            "n_clips": len(list(b["out"].glob("*.npy"))), "n_failed": b["fail"],
            "layout": f"0:{D} backbone | {D} motion_inst | {D+1} motion_disp",
            "frames": "same dense extraction + same frames_used indices as the CLIP build",
            "motion": "copied from the CLIP build" if reuse else "recomputed"}, indent=1))
        print(f"wrote {b['out']}  dim={D}  arrays={len(list(b['out'].glob('*.npy')))}  "
              f"failed={b['fail']}")
    print(f"\ntotal {seen} clips in {(time.time()-t0)/60:.1f} min "
          f"({seen/max(time.time()-t0,1e-9)*60:.0f} clips/min)")


if __name__ == "__main__":
    main()
