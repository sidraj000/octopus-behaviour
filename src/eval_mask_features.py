"""eval_mask_features.py — do segmentation-derived features improve the ethogram classifier?

ONE comparison, held as tight as possible: the same backbone, rung, splits, seeds and loss, with and
without the 10 mask-geometry channels from `extract_mask_feats.py` appended. Nothing else moves.

WHAT THIS REPORTS, and why each part is needed rather than nice-to-have:

  1. MACRO-F1 with vs without, val-selected, 3 seeds. The headline.

  2. PER-CLASS deltas. The prediction is specific -- mask area should help `No octopus` vs `Resting`
     (22% of all errors, and their whole-frame motion medians are identical so the existing channels
     provably cannot separate them), centroid displacement should help `Locomotion`, centroid height
     should help `Reaching`. If macro-F1 rises but those three do not, the gain is not coming from the
     mechanism claimed and should not be described as if it were.

  3. SPLIT BY seg_seen_video. The segmenter trained on 11 of the 34 ethogram test videos, so its
     features are sharper there than in deployment. Pooling the two would launder that advantage into
     the headline. The UNSEEN-video subset is the honest number; both are printed.

  4. AN IR CONTROL. IR clips carry a zeroed mask block by construction, so they are a built-in
     negative control: if the "gain" appears on IR too, it is not coming from the masks and something
     is wrong with the comparison.

  5. THE NO-OCTOPUS/RESTING CONFUSION COUNT specifically, before and after -- the single number this
     experiment was designed to move.

Usage: venv/bin/python3 src/eval_mask_features.py [--backbone videomae] [--rung 2]
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import train_ethogram as T
from extract_mask_feats import seg_training_videos, N_CH

MASKDIR = None      # set in main once the version is known


def load_mask_block(clips):
    """clip -> [T, 10] mask features. Missing files are reported, not silently zero-filled."""
    out, missing = {}, []
    for k in clips:
        f = MASKDIR / (k.replace("/", "__") + ".npy")
        if f.exists():
            out[k] = np.load(f).astype(np.float32)
        else:
            missing.append(k)
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="videomae")
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "data" / "ethogram_mask_features.json"))
    a = ap.parse_args()

    global MASKDIR
    MASKDIR = REPO / "src" / "dataset_etho" / a.version / "feats_mask"
    man, X, classes, D = T.load(a.version, a.backbone)
    mask, missing = load_mask_block([r["clip"] for r in man])
    if missing:
        print(f"WARNING {len(missing)} clips have no mask features -- EXCLUDED from BOTH arms so the "
              f"comparison stays on identical rows")
        man = [r for r in man if r["clip"] not in set(missing)]
    T_len = next(iter(mask.values())).shape[0]
    print(f"mask block: {len(mask)} clips, [{T_len}, {N_CH}]")

    seen_vids = seg_training_videos()
    te = [r for r in man if r["split"] == "test"]
    n_seen = sum(1 for r in te if r["video"] in seen_vids)
    print(f"test clips: {len(te)}  from videos the segmenter SAW: {n_seen} "
          f"({len({r['video'] for r in te if r['video'] in seen_vids})} videos), "
          f"UNSEEN: {len(te)-n_seen}")
    valid_frac = np.mean([float(mask[r["clip"]][:, 9].mean()) for r in te])
    ir = [r for r in te if r["camera"] == "Right_Top"]
    print(f"mean valid-frame fraction on test: {valid_frac:.2f}   IR test clips (zeroed block): {len(ir)}")

    # ---- the two arms ----
    results = {"backbone": a.backbone, "rung": a.rung, "cw_power": T.CW_POWER, "arms": {}}
    preds = {}
    for arm in ("without_mask", "with_mask"):
        if arm == "with_mask":
            # Append the mask block along the FEATURE axis. Sequence lengths can differ (VideoMAE
            # emits T=8, the mask block follows the 10 sampled frames), so resample the mask block
            # to the backbone's T by nearest index -- no interpolation, so no invented values.
            Xa = {}
            for r in man:
                b = X[r["clip"]]; m = mask[r["clip"]]
                if m.shape[0] != b.shape[0]:
                    idx = np.linspace(0, m.shape[0] - 1, b.shape[0]).round().astype(int)
                    m = m[idx]
                Xa[r["clip"]] = np.concatenate([b[:, :D], m, b[:, D:]], 1).astype(np.float32)
            Da = D + N_CH
        else:
            Xa, Da = X, D
        vs, ts, per_cls, pr = [], [], collections.defaultdict(list), None
        for s in range(a.seeds):
            o = T.run_one(a.rung, man, Xa, classes, s, D=Da)
            vs.append(o["val_f1"]); ts.append(o["test_f1"])
            for ci in range(len(classes)):
                per_cls[classes[ci]].append(o["per_class"][ci]["f1"])
            pr = np.asarray(o["pred"]) if pr is None else pr
        preds[arm] = pr
        results["arms"][arm] = {
            "val": round(float(np.mean(vs)), 4), "val_std": round(float(np.std(vs)), 4),
            "test": round(float(np.mean(ts)), 4), "test_std": round(float(np.std(ts)), 4),
            "per_class": {c: round(float(np.mean(v)), 4) for c, v in per_cls.items()},
            "feat_dim": Da}
        r_ = results["arms"][arm]
        print(f"\n{arm:<14} D={Da:<5} val {r_['val']:.4f} ±{r_['val_std']:.4f}   "
              f"TEST {r_['test']:.4f} ±{r_['test_std']:.4f}")

    w, wo = results["arms"]["with_mask"], results["arms"]["without_mask"]
    dv, dt = w["val"] - wo["val"], w["test"] - wo["test"]
    print(f"\ndelta: val {dv:+.4f} (with-mask seed std {w['val_std']:.4f})   TEST {dt:+.4f}")
    verdict = ("REAL val gain" if dv > w["val_std"] else
               "inside seed noise -- mask features do NOT help")
    print(f"  -> {verdict}")
    results["delta"] = {"val": round(dv, 4), "test": round(dt, 4), "verdict": verdict}

    # ---- per-class, where the mechanism claim lives ----
    print(f"\n{'class':<34}{'without':>9}{'with':>9}{'delta':>9}   prediction")
    pred_for = {"No octopus": "area separates it from Resting",
                "Resting / stationary": "area separates it from No-octopus",
                "Locomotion (crawl/swim)": "centroid displacement",
                "Reaching out of water": "centroid height",
                "Human / enrichment interaction": "(no mechanism -- control)",
                "Exploration / manipulation": "(weak -- posture only)"}
    for c in classes:
        b_, x_ = wo["per_class"][c], w["per_class"][c]
        print(f"  {c:<32}{b_:>9.3f}{x_:>9.3f}{x_-b_:>+9.3f}   {pred_for.get(c,'')}")

    # ---- the confusion this was designed to fix ----
    Hte = np.array([r["label_idx"] for r in te])
    ia, ir_i = classes.index("No octopus"), classes.index("Resting / stationary")
    print(f"\nNo-octopus <-> Resting confusions on test (the target of this experiment):")
    for arm in ("without_mask", "with_mask"):
        p = preds[arm]
        n = int(sum(1 for a_, b_ in zip(Hte, p)
                    if (a_ == ia and b_ == ir_i) or (a_ == ir_i and b_ == ia)))
        err = int((p != Hte).sum())
        print(f"  {arm:<14} {n} of {err} total errors ({n/max(err,1):.1%})")
        results["arms"][arm]["static_confusions"] = n

    # ---- seen vs unseen segmenter videos, and the IR control ----
    print(f"\nsplit by whether the SEGMENTER saw the test video (unseen = the honest number):")
    for name, sel in (("seg SAW the video", lambda r: r["video"] in seen_vids),
                      ("seg UNSEEN", lambda r: r["video"] not in seen_vids),
                      ("IR only (zeroed block = control)", lambda r: r["camera"] == "Right_Top")):
        idx = [i for i, r in enumerate(te) if sel(r)]
        if not idx:
            continue
        row = {}
        for arm in ("without_mask", "with_mask"):
            p = preds[arm][idx]; t = Hte[idx]
            f1, _ = T.macro_f1(p, t, len(classes))
            row[arm] = (f1, float((p == t).mean()))
        d = row["with_mask"][0] - row["without_mask"][0]
        print(f"  {name:<34} n={len(idx):<5} macroF1 {row['without_mask'][0]:.4f} -> "
              f"{row['with_mask'][0]:.4f} ({d:+.4f})   acc {row['without_mask'][1]:.3f} -> "
              f"{row['with_mask'][1]:.3f}")
        results.setdefault("subsets", {})[name] = {
            "n": len(idx), "macro_f1_without": round(row["without_mask"][0], 4),
            "macro_f1_with": round(row["with_mask"][0], 4), "delta": round(d, 4)}
    print("\n  If the IR row (zeroed mask block) shows a gain comparable to the others, the effect is\n"
          "  NOT coming from the masks and the comparison is confounded.")
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
