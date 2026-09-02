"""make_release.py — assemble everything for upload in one folder: models, datasets, docs.

Produces `release/` ready to drop on Drive:

    release/
      README.md              what this is, what is in it, what the caveats are
      models/                clean public names + MODELS.md cards + manifest.json
      datasets/              labels, captions, benchmarks, frozen dataset snapshot
      docs/                  the paper PDF and the experimental record

WHAT IS DELIBERATELY NOT HERE. The 20 s video clips (~46 GB). Every label file carries the
`video_url` and byte range it came from, so the clips are reconstructible from the source archive
without shipping them -- and shipping footage of a captive animal is a permission question, not a
storage one.

Sizes are checked against free disk BEFORE copying, because this machine has repeatedly run low and
a half-copied release is worse than none.

Usage:
  venv/bin/python3 src/make_release.py --dry-run     # report what would be staged
  venv/bin/python3 src/make_release.py               # stage it
"""
import os
import argparse, json, shutil, subprocess, sys
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



def _consolidate_feats(backbone, dst):
    """4,665 individual .npy -> one fp16 .npz. Halves the download and is one file to load."""
    import numpy as np
    d = REPO / "src" / "dataset_etho" / "v1" / f"feats_{backbone}"
    if not d.exists():
        return None
    arrs = {}
    for f in sorted(d.glob("*.npy")):
        arrs[f.stem.replace("__", "/")] = np.load(f).astype(np.float16)
    if not arrs:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrs)
    return dst.stat().st_size

# (source, destination-relative, what it is)  -- directories are copied whole
DATASETS = [
    ("data/ensemble_235b_voted.json", "labels/teacher_votes_5pass.json",
     "5,222 clips with the FULL 5-pass vote distribution for presence and behaviour, not just the "
     "majority. The margin predicts human agreement, which is why the distribution is kept."),
    ("data/ensemble_235b", "labels/teacher_passes",
     "The five raw passes: ~25,000 captions (5,160 clips x 5 disjoint frame samplings) plus each "
     "pass's presence and behaviour call. Lets caption stability under input perturbation be "
     "measured, which one caption per clip cannot."),
    ("src/dataset_etho/v1/manifest.jsonl", "ethogram_dataset_v1/manifest.jsonl",
     "The frozen 6-class dataset: 4,665 clips / 204 videos, one row per clip with its soft target, "
     "split assignment, per-clip reliability weight and motion summary."),
    ("src/dataset_etho/v1/snapshot.json", "ethogram_dataset_v1/snapshot.json",
     "Class list, split sizes by clip AND by video, merge rules, and the camera-directional filter."),
    ("src/dataset_etho/v1/human_secondary.jsonl", "ethogram_dataset_v1/human_secondary.jsonl",
     "The human-labelled clips held out of train/val, with both the human and teacher label."),
    ("src/dataset_etho/v1/features.npz", "ethogram_dataset_v1/features_clip.npz",
     "Cached CLIP features [10, 514] per clip so the classifier can be retrained without a GPU or "
     "the video. Regenerable by src/extract_backbone_feats.py."),
    # The four other backbones' features. Without these the paper's HEADLINE model (the five-member
    # ensemble at 0.665 macro-F1) cannot be reproduced at all -- only the CLIP-only rungs can. They
    # are derived from the clips but are not clips, so they carry no footage-permission question and
    # make the headline result reproducible with NO video access whatsoever. Consolidated to one npz
    # per backbone at fp16: these are inputs to a small MLP, so fp16 is lossless in effect and halves
    # the download.
    ("__FEATS__dinov2", "ethogram_dataset_v1/features_dinov2.npz",
     "DINOv2-base features [10, 770] per clip (768 backbone + 2 motion channels), fp16."),
    ("__FEATS__videomae", "ethogram_dataset_v1/features_videomae.npz",
     "VideoMAE-base features [8, 770] per clip, fp16. T=8 because video models emit one token per "
     "temporal position rather than per sampled frame."),
    ("__FEATS__dinov2crop", "ethogram_dataset_v1/features_dinov2_maskcrop.npz",
     "DINOv2 on mask-guided crops, fp16. Cropping to the animal was worth +0.07 macro-F1."),
    ("__FEATS__videomaecrop", "ethogram_dataset_v1/features_videomae_maskcrop.npz",
     "VideoMAE on mask-guided crops, fp16."),
    ("data/human_behaviour_labels.json", "human_labels/behaviour_round1.json", "Human behaviour labels, round 1."),
    ("data/human_behaviour_labels_v2.json", "human_labels/behaviour_round2.json", "Round 2."),
    ("data/human_behaviour_labels_v3.json", "human_labels/behaviour_round3_ir_presence.json",
     "Round 3, targeting infrared presence, with a colour control arm."),
    ("data/human_eval_sample_v1.json", "human_labels/sample_round1.json", "The frozen sample round 1 drew from."),
    ("data/human_eval_sample_v2.json", "human_labels/sample_round2.json", "Round 2's sample."),
    ("data/human_eval_sample_v3.json", "human_labels/sample_round3.json", "Round 3's sample, incl. group labels."),
    # Segmentation mask data is NOT staged here -- the author already holds it. The mask MODEL and
    # the SEG-TEST results are still in the release; only the 49 MB of mask images/labels are omitted.
    ("data/benchmarks.json", "benchmarks/results.json",
     "Frozen suite results: SEG-TEST (122 human masks / 5 held-out videos + 19 empty-tank "
     "negatives), SKEL-50 (50 human arm-keypoint frames / 20 videos), and fusion variants."),
    ("data/behaviour_records.json", "behaviour/structured_records.json",
     "3,205 clips x 9 structured fields (behaviour, posture, activity, location, context, colour)."),
    ("data/behaviour_stats.json", "behaviour/aggregate_stats.json",
     "Activity budget, exposure-normalised circadian profile, stimulus response by context."),
    ("data/skeleton_motion.json", "behaviour/skeleton_kinematics.json",
     "Per-clip arm kinematics: tip/mantle speed, arm spread, occluded fraction."),
    ("data/harvest_ledger_all.json", "coverage/harvest_ledger.json",
     "Per-video coverage over 1,769 videos: probe scores, seconds scanned AND unscanned, and why "
     "each video was kept or discarded. The only record of what was looked at."),
    ("data/harvest_clips_index.json", "coverage/clips_index.json",
     "Clip provenance: source video_url, camera, date, and the byte range each clip came from."),
    ("src/ethogram_list_v2.json", "ethogram_dataset_v1/class_sheet.json",
     "The 7-class ethogram sheet, with the maps_from folding from the original 19."),
    ("data/zeroshot_vs_probe.json", "benchmarks/zeroshot_vs_probe.json",
     "The detector-independent presence comparison: 120 frames at uniform random timestamps."),
]

DOCS = [
    ("PAPER_NOTES.md", "experimental_record.md",
     "The full chronological ledger, R1-R35, INCLUDING the failed experiments and the conclusions "
     "that were retracted."),
    ("RESULTS_ETHOGRAM.md", "ethogram_results.md", "Tidied current state of the ethogram work."),
    ("src/SEGMENTATION_LOG.md", "segmentation_log.md", "The segmentation trail."),
]

README = """# Octopus behaviour pipeline — models and datasets

Models, labels and the experimental record for turning raw aquarium video into a behavioural
time series for a captive octopus.

    models/     five deployable models + a card each (MODELS.md)
    datasets/   labels, captions, benchmarks, frozen dataset snapshot
    docs/       the full experimental record

## Read these three things first

**1. The behaviour labels are TEACHER labels.** All 5,222 clip labels come from a 5-pass
Qwen3-VL-235B ensemble, not from a human. The full vote distribution is kept, not just the
majority, because the margin predicts human agreement (0.73 unanimous / 0.86 at four-of-five /
0.43 at three-or-fewer). Treat them as a strong automatic annotation, not ground truth.

**2. The human labels measure AGREEMENT, not accuracy.** All 456 human behaviour labels were
collected with the model's suggestion visible on screen. Every label records an `assisted` flag,
and every one is true. A blind round is specified but was not run before release. The human MASKS
(513) and the arm keypoints were drawn independently and do not carry this caveat.

**3. One animal, one tank.** Everything is a single adult *Octopus vulgaris*. Cross-animal
generalisation is untested.

## Protocol rules the data was built under

1. Splits are by **source video**, never by clip.
2. Frozen evaluation sets are never regenerated to suit a result.
3. Negatives of different kinds (reflections / empty tank / infrared) are never pooled.
4. Holdout videos are excluded from **every** training source, not just the final stage.

## The video clips are not here

~46 GB of 20 s clips are not included. `datasets/coverage/clips_index.json` carries each clip's
source `video_url`, camera, date and byte range, so clips are reconstructible from the source
archive. Shipping footage of a captive animal is a permission question rather than a storage one.

## Licence

TO BE SET before distribution. Suggested: code Apache-2.0, data CC-BY-4.0.

## Citation

TO BE ADDED once the paper has a DOI.
"""


def size_of(p):
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def free_bytes(path):
    st = shutil.disk_usage(path)
    return st.free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "release"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    plan, missing, total = [], [], 0
    for src, dst, why in DATASETS:
        if src.startswith("__FEATS__"):
            b = src.replace("__FEATS__", "")
            d = REPO / "src" / "dataset_etho" / "v1" / f"feats_{b}"
            if not d.exists():
                missing.append(f"feats_{b}"); continue
            s = int(size_of(d) * 0.5)          # fp16 + compression, measured ~0.5x
            total += s
            plan.append(("datasets/" + dst, src, s, why))
            continue
        p = REPO / src
        if not p.exists():
            missing.append(src); continue
        s = size_of(p); total += s
        plan.append(("datasets/" + dst, p, s, why))
    for src, dst, why in DOCS:
        p = REPO / src
        if not p.exists():
            missing.append(src); continue
        s = size_of(p); total += s
        plan.append(("docs/" + dst, p, s, why))

    # models come from the model-release manifest so the two cannot disagree
    mm = REPO / "release" / "models" / "manifest.json"
    model_total = 0
    if mm.exists():
        man = json.load(open(mm))
        for name, e in man["models"].items():
            model_total += e["bytes"]
    total += model_total

    print(f"{'destination':<52}{'size':>10}")
    for dst, _p, s, _w in plan:
        print(f"  {dst:<50}{s/1e6:>9.1f} MB")
    print(f"  {'models/ (from manifest.json)':<50}{model_total/1e6:>9.1f} MB")
    print(f"\nTOTAL {total/1e9:.2f} GB     free disk {free_bytes(REPO)/1e9:.2f} GB")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for m in missing:
            print(f"  {m}")

    if a.dry_run:
        print("\n[dry-run] nothing written.")
        return
    if total * 1.15 > free_bytes(REPO):
        sys.exit(f"\nREFUSING: need ~{total*1.15/1e9:.2f} GB with headroom, "
                 f"only {free_bytes(REPO)/1e9:.2f} GB free. A half-copied release is worse than "
                 f"none -- free space or pass --out to another volume.")

    for dst, p, _s, _w in plan:
        d = out / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(p, str) and p.startswith("__FEATS__"):
            b = p.replace("__FEATS__", "")
            got = _consolidate_feats(b, d)
            print(f"  consolidated feats_{b} -> {d.name} ({(got or 0)/1e6:.1f} MB)")
        else:
            _place(p, d)
    print("\ncopying models via make_model_release.py --copy ...")
    subprocess.run([sys.executable, str(REPO / "src" / "make_model_release.py"),
                    "--out", str(out / "models"), "--copy"], check=True)

    (out / "README.md").write_text(README)
    inv = {"datasets": {dst.split("/", 1)[1]: {"bytes": s, "what": w}
                        for dst, _p, s, w in plan if dst.startswith("datasets/")},
           "docs": {dst.split("/", 1)[1]: {"bytes": s, "what": w}
                    for dst, _p, s, w in plan if dst.startswith("docs/")}}
    (out / "datasets" / "CONTENTS.json").write_text(json.dumps(inv, indent=1))
    print(f"\nstaged {out}  ({size_of(out)/1e9:.2f} GB) -- ready to upload")


if __name__ == "__main__":
    main()
