"""consolidate_seg_data.py — gather every segmentation dataset into one self-describing folder.

The segmentation data ended up scattered across a Modal volume, a Downloads folder and the repo,
under names that do not say how they relate. Measured relationships, which are not obvious:

    v1  (4,412 pairs)  is 100% INSIDE v3        -> redundant, not copied
    human (502 images) is 100% INSIDE thin768   -> images not copied, manifest kept
    thin768 vs v3      share ZERO filenames     -> genuinely different image sets

So only two sets carry unique pixels: `thin768` (the training set of the RELEASED model) and `v3`
(the earlier auto-label run whose 1,388 negatives fixed the presence gate). Copying v1 and the human
images as well would add ~500 MB of duplicates and invite someone to train on the same frames twice.

Files are HARDLINKED, not copied: everything is on one volume, so the consolidated folder costs
almost no disk while still uploading as ordinary files.

Usage: venv/bin/python3 src/consolidate_seg_data.py [--out data/segmentation_all] [--dry-run]
"""
import argparse, json, os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/Users/siddharthraj/Downloads/data 2/dataset_seg")

SETS = {
    "thin768_released": {
        "src": REPO / "data" / "dataset_seg_thin768",
        "what": "Training set of the RELEASED model (octo-mask-lraspp-3.2M-v1). 5,244 manifest rows "
                "over 188 source videos: 4,731 HQ auto-labelled (GroundingDINO-base + SAM2-large), "
                "412 human positives, 87 human empty-mask negatives, 14 marked reject.",
        "images": "uniformly 1024x576",
        "note": "Colour cameras only (Right_Back / Right_Front / Right_Right). No infrared, which "
                "is why the released model must not be used on IR footage.",
    },
    "v3_autolabel": {
        "src": DOWNLOADS / "v3",
        "what": "Earlier auto-label run, 5,800 pairs over 94 videos = 4,412 positives + 1,388 "
                "empty-mask negatives. The negatives are the result that mattered: adding them "
                "took the presence gate from AUC 0.50 (a coin flip that fired on reflections) to "
                "0.86 overall and 0.99 against reflections.",
        "images": "positives 1024x576; the 1,366 `neg_*` negatives are FULL 4K (3840x2160), which "
                 "is why this set is 1.4 GB rather than ~550 MB. They are downsampled at training "
                 "time, so the extra pixels buy nothing.",
        "note": "Includes Right_Left (800 pairs), the reflection camera. Supersedes v1 entirely.",
    },
}

# Manifests worth keeping even where the images are duplicates elsewhere.
MANIFEST_ONLY = {
    "human_masks_manifest.jsonl": {
        "src": REPO / "data" / "dataset_seg_human" / "manifest.jsonl",
        "what": "The 513 human click-to-SAM2 masks (412 positive + 87 negative + 14 reject) with "
                "their per-mask area and seed frame. All 502 images are already inside "
                "thin768_released, so only the manifest is here -- but this is the file that says "
                "WHICH of those are human-drawn rather than auto-labelled.",
    },
}

SKIPPED = {
    "v1": "4,412 pairs, 100% contained in v3_autolabel (verified by filename). Copying it would "
          "duplicate ~450 MB and invite training on the same frames twice.",
    "dataset_seg_human/images": "All 502 images are inside thin768_released. The manifest is kept.",
}

README = """# Segmentation data — consolidated

Two sets carry unique pixels. Their relationship is not obvious from the names, so it is stated
here and was verified by filename comparison rather than assumed:

    v1                is 100% INSIDE v3          -> v1 is not included
    human masks (502) are 100% INSIDE thin768    -> only the manifest is included
    thin768 and v3    share ZERO filenames       -> different image sets, both kept

## `thin768_released/` — {t_size}

{t_what}

Images: {t_images}
{t_note}

**This is the set behind the published checkpoint.** Retraining from `v3_autolabel` will not
reproduce IoU 0.6415.

## `v3_autolabel/` — {v_size}

{v_what}

Images: {v_images}
{v_note}

## `human_masks_manifest.jsonl`

{h_what}

## Not included

{skipped}

## Layout

Each set is `images/`, `masks/` and `manifest.jsonl`. Manifest rows carry the source clip, camera,
mask area, and for thin768 also the SAM2 prompt that produced the mask (`seed_frame`, `points`,
`labels`, `box`) — so the masks are regenerable, not only redistributable.
"""


def link_tree(src, dst):
    """Hardlink a tree. Same volume, so the consolidated folder costs almost no disk."""
    n = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        d = dst / f.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            d.unlink()
        try:
            os.link(f, d)
        except OSError:
            shutil.copy2(f, d)
        n += 1
    return n


def du(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "segmentation_all"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    missing = [k for k, v in SETS.items() if not v["src"].exists()]
    if missing:
        sys.exit(f"missing source sets: {missing}")

    inv, total = {}, 0
    for name, meta in SETS.items():
        s = du(meta["src"]); total += s
        n_img = len(list((meta["src"] / "images").glob("*")))
        n_msk = len(list((meta["src"] / "masks").glob("*")))
        inv[name] = {"bytes": s, "images": n_img, "masks": n_msk,
                     **{k: v for k, v in meta.items() if k != "src"}}
        print(f"  {name:<20} {n_img:>6} img / {n_msk:>6} mask   {s/1e6:>8.1f} MB")
    print(f"\n  TOTAL {total/1e9:.2f} GB (hardlinked, so ~0 extra disk)")
    if a.dry_run:
        print("\n[dry-run] nothing written."); return

    for name, meta in SETS.items():
        n = link_tree(meta["src"], out / name)
        print(f"  linked {n} files -> {name}/")
    for fname, meta in MANIFEST_ONLY.items():
        if meta["src"].exists():
            shutil.copy2(meta["src"], out / fname)
            inv[fname] = {"bytes": meta["src"].stat().st_size, "what": meta["what"]}
            print(f"  copied {fname}")

    t, v = inv["thin768_released"], inv["v3_autolabel"]
    (out / "README.md").write_text(README.format(
        t_size=f"{t['bytes']/1e6:.0f} MB, {t['images']} pairs", t_what=t["what"],
        t_images=t["images"], t_note=t["note"],
        v_size=f"{v['bytes']/1e6:.0f} MB, {v['images']} pairs", v_what=v["what"],
        v_images=v["images"], v_note=v["note"],
        h_what=MANIFEST_ONLY["human_masks_manifest.jsonl"]["what"],
        skipped="\n".join(f"- **{k}** — {why}" for k, why in SKIPPED.items())))
    (out / "MANIFEST.json").write_text(json.dumps({"sets": inv, "skipped": SKIPPED}, indent=1))
    print(f"\nwrote {out}  ({du(out)/1e9:.2f} GB apparent)")


if __name__ == "__main__":
    main()
