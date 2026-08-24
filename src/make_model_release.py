"""make_model_release.py — assemble the published model set under clean, self-describing names.

WHY NOT JUST RENAME THE FILES. 21 files reference `clip_mlp_hardneg_v2.pt` and 17 reference
`octo_seg_thin768_lraspp.pt`; the project record is explicit about which script loads which checkpoint and
warns against changing a script's active model without instruction. Renaming in place would touch
~40 call sites to make a cosmetic improvement, and a missed one fails at runtime on a path string.
So the working tree keeps its names and the RELEASE gets good ones, with the mapping recorded here
and in the generated MODELS.md.

NAMING SCHEME  octo-<task>-<arch>[-<detail>]-v<n>
Task first, because that is how someone browsing a release looks for a model -- not by the internal
experiment tag that happened to win a sweep. `thin768`, `clean512tv` and `hardneg_v2` are sweep
labels that mean nothing outside this repo, and one of them ("hardneg") refers to a step the paper
no longer describes.

Every model gets a card in MODELS.md carrying what it was trained on, what it scores on which frozen
set, and what it must NOT be used for. The caveats are the point: two of these models reproduce a
VLM teacher rather than ground truth, and one is unusable on 35% of the corpus.

Usage: venv/bin/python3 src/make_model_release.py [--out release/models]
"""
import os
import argparse, hashlib, json, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def _place(src, dst):
    """Hardlink if possible, else copy. The MLX model alone is 1.8 GB; duplicating it to stage a
    release wastes disk this machine does not have, and a hardlink is identical for upload."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for f in src.rglob("*"):
            if f.is_file():
                d = dst / f.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(f, d)
                except OSError:
                    shutil.copy2(f, d)
    else:
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


# public name -> (source path, card)
MODELS = {
    "octo-presence-clip-probe-v1.pt": {
        "src": "weights/clip_mlp_hardneg_v2.pt",
        "task": "Presence gate: is an octopus visible in this frame?",
        "arch": "frozen CLIP ViT-B/32 + MLP probe (512-256-64-2), 0.6 MB",
        "trained_on": "~11k human-labelled frames over 8 dates, letterbox preprocessing, "
                      "including 66 human-verified hard negatives",
        "scores": "AUC 0.745 on 120 human-labelled frames sampled at uniform random timestamps "
                  "over 60 videos (detector-independent). Zero-shot CLIP on the same frames: 0.450.",
        "use_for": "first-stage gating of long footage before an expensive model",
        "do_not": "do not read the 96.8% training accuracy as field performance -- that was a "
                  "different, easier test set. Preprocessing MUST be letterbox, not centre crop.",
    },
    "octo-mask-lraspp-3.2M-v1.pt": {
        "src": "weights/seg/octo_seg_thin768_lraspp.pt",
        "task": "Single-class octopus segmentation, no prompt at inference",
        "arch": "LR-ASPP over MobileNetV3, 3.2 M parameters, 768x768 input",
        "trained_on": "auto-labelled GroundingDINO+SAM2 masks blended with 513 human masks; "
                      "Focal-Tversky loss; test videos excluded from every training source",
        "scores": "IoU 0.6415 mean / 0.7193 median and mean AREA ERROR ~1% against 122 human masks "
                  "on 5 held-out videos. The zero-shot teacher scores 0.374 per-frame on the same "
                  "masks (paired delta -0.268, CI95 [-0.313,-0.136]).",
        "use_for": "presence, body area as a posture proxy, motion inside the mask, and as the "
                   "input to skeletonisation",
        "do_not": "do not use on infrared footage -- it is colour-trained and over-segments bright "
                  "metal tools there (35% of our corpus is IR and is excluded downstream). Do not "
                  "expect boundary-accurate masks; area is accurate, thin arms are not.",
    },
    "octo-ethogram-ensemble-v1.pt": {
        "src": "weights/ethogram_ensemble_v1.pt",
        "task": "6-class behaviour classification, absence included as a class",
        "arch": "5 frozen backbones (CLIP, DINOv2, VideoMAE, + two mask-cropped views), one "
                "pooled-statistics MLP head each, 3 seeds per member, soft vote. 33.5 MB",
        "trained_on": "4,665 clips / 204 videos labelled by a 5-pass Qwen3-VL-235B ensemble, with "
                      "SOFT targets from the vote distribution; splits by source video",
        "scores": "macro-F1 0.6648 / accuracy 0.7541 on 740 test clips from 34 held-out videos. "
                  "Majority-class baseline: 0.1004 / 43.1%.",
        "use_for": "assembling a behavioural time series from extracted clips",
        "do_not": "this reproduces the VLM teacher, NOT human ground truth. On the same clips the "
                  "teacher agrees with a human 72.5% and this student 66.9%, and every human label "
                  "used for that comparison was collected with the model's answer on screen "
                  "(agreement, not accuracy). Per-class F1 for human/enrichment interaction rests "
                  "on 40 clips from 7 videos and is not reliable.",
    },
    "octo-caption-qwen3vl2b-lora-v1": {
        "src": "models/qwen3vl2b_caption_v1_lora",
        "task": "One-sentence behaviour caption for a 20 s clip",
        "arch": "Qwen3-VL-2B + LoRA r16/alpha32 adapter, 77 MB (needs the base model)",
        "trained_on": "3,066 teacher captions from Qwen3-VL-235B, on CLAHE-enhanced best-N frames",
        "scores": "held-out embedding similarity 0.834 (base 0.702), ROUGE-L 0.455 (base 0.269)",
        "use_for": "captioning where a GPU is available",
        "do_not": "frame preparation matters more than the model here -- feeding uniform, "
                  "unenhanced frames was preferred over enhanced ones in only 1 of 33 blind "
                  "comparisons. Use the same CLAHE + best-frame selection it was trained with.",
    },
    "octo-caption-qwen3vl2b-4bit-v1": {
        "src": "models/qwen3vl2b_caption_v1_mlx_4bit",
        "task": "Same, quantised for on-device use",
        "arch": "Qwen3-VL-2B, 4-bit MLX, 1.7 GB, self-contained",
        "trained_on": "as above, then merged and quantised",
        "scores": "~3 s per clip on a 16 GB Apple Silicon laptop, no GPU, no API",
        "use_for": "running the caption stage on site",
        "do_not": "MLX is Apple Silicon only; bitsandbytes NF4 is the CUDA equivalent and is not "
                  "interchangeable with this file.",
    },
}

# Deliberately NOT released, with the reason recorded so the omission is not read as an oversight.
WITHHELD = {
    "weights/clip_mlp_v3.pt":
        "Retrained presence gate. Large improvement on diverse footage (FPR 0.485 -> 0.243, AUC "
        "0.777 -> 0.906) but it FORGOT its original domain: recall on held-out anchor frames fell "
        "0.985 -> 0.903 and AUC 0.999 -> 0.924. Training was 68% new-domain, so the boundary "
        "shifted. Withheld until the anchor regression is fixed by rebalancing the two domains.",
    "weights/seg/octo_seg_clean512tv_lraspp.pt":
        "512-input segmentation variant, superseded on every metric by the 768 model "
        "(IoU 0.608 vs 0.642, presence AUC 0.718 vs 0.794). Kept for the resolution ablation only.",
    "weights/clip_mlp_best.pt":
        "Prior presence probe, superseded. Trained on 66 hard negatives instead of ~1.7k.",
}


def sha256(p):
    h = hashlib.sha256()
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(f.relative_to(p).as_posix().encode())
            h.update(f.read_bytes())
    else:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "release" / "models"))
    ap.add_argument("--copy", action="store_true", help="copy files (default: manifest only)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    manifest, missing = {}, []
    lines = ["# Released models\n",
             "Names follow `octo-<task>-<arch>-v<n>`. The working repository uses different, older "
             "filenames (sweep labels such as `thin768` or `hardneg_v2`); the mapping is in each "
             "card below and in `manifest.json`. Nothing was renamed in place because ~40 call "
             "sites reference the old paths.\n"]
    for name, m in MODELS.items():
        src = REPO / m["src"]
        if not src.exists():
            missing.append(m["src"]); continue
        size = (sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
                if src.is_dir() else src.stat().st_size)
        digest = sha256(src)
        manifest[name] = {**{k: v for k, v in m.items() if k != "src"},
                          "internal_path": m["src"], "bytes": size, "sha256_16": digest}
        if a.copy:
            dst = out / name
            _place(src, dst)
        lines += [f"\n## `{name}`\n",
                  f"- **Task** {m['task']}",
                  f"- **Architecture** {m['arch']}",
                  f"- **Trained on** {m['trained_on']}",
                  f"- **Measured** {m['scores']}",
                  f"- **Use for** {m['use_for']}",
                  f"- **Do not** {m['do_not']}",
                  f"- Internal path `{m['src']}` · {size/1e6:.1f} MB · sha256 `{digest}`"]

    lines += ["\n---\n\n## Deliberately withheld\n",
              "Recorded so the omissions are not read as oversights.\n"]
    for p, why in WITHHELD.items():
        lines.append(f"- **`{Path(p).name}`** — {why}")

    (out / "MODELS.md").write_text("\n".join(lines) + "\n")
    (out / "manifest.json").write_text(json.dumps(
        {"models": manifest, "withheld": WITHHELD}, indent=1))
    print(f"wrote {out/'MODELS.md'} and manifest.json  ({len(manifest)} models"
          + (", files copied" if a.copy else ", manifest only -- pass --copy to stage files") + ")")
    for name, e in manifest.items():
        print(f"  {name:<40} {e['bytes']/1e6:>8.1f} MB   <- {e['internal_path']}")
    if missing:
        print("\nMISSING sources (not written to the manifest):")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
