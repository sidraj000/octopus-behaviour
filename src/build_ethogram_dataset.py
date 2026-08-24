"""build_ethogram_dataset.py — freeze the training set for the 6-class ethogram classifier.

Turns the 5-pass 235B ensemble into a trainable dataset: filtered by MEASURED label reliability,
merged, split by SOURCE VIDEO, with SOFT targets from the vote distribution and cached per-frame
CLIP features.

--------------------------------------------------------------------------------------------------
CLASSES (6). Behaviour AND absence in one head, because that is how it is deployed: a clip arrives,
is there an animal, and what is it doing. Precedent: the segmentation work found that training WITH
negatives turned a coin-flip presence gate (AUC 0.50) into 0.86.
    No octopus · Resting/stationary · Exploration/manipulation ·
    Locomotion (crawl/swim) · Reaching out of water · Human/enrichment interaction

MERGE Crawling + Swimming/jetting -> one locomotion class. The model cannot separate them (5 of v2's
40 behaviour errors were Swimming->Crawling, all one direction) and Swimming had 42 clips in 14
videos, too few for a per-class F1 under a video-level split. TRAINING-TIME MAPPING ONLY: the
extraction prompt, ethogram_list_v2.json and every stored record keep the 7-class vocabulary, so
R15's kappa and the human rounds stay comparable and the merge is reversible.
Applied to the VOTE DISTRIBUTION too -- 3 Crawling / 2 Swimming becomes unanimous locomotion, which
is right: the ensemble was certain about locomotion, just not which kind.

DROP `Colour change / defensive`: 1 clip corpus-wide. Unlearnable.

--------------------------------------------------------------------------------------------------
CAMERA-DIRECTIONAL FILTERING -- the important part, and it is measured, not assumed.
From 298 human labels, label reliability depends on the camera AND the direction:

    camera        model says ABSENT -> human agrees      model says PRESENT
    Right_Left    36/36 = 100%                           WRONG 45% of the time
    Right_Right    8/8  = 100%                           0% FP
    Right_Front   18/20 =  90%                           3% FP
    Right_Back     7/8  =  88%                          15% FP
    Right_Top     11/20 =  55%   <- worst                 8% FP

Two opposite failure modes, both physically sensible: Right_Left HALLUCINATES presence (tank-glass
reflections) but is perfect when it says absent; Right_Top (IR) MISSES animals in dim footage but is
reliable when it does see one. So the filter is directional rather than a blanket camera exclusion --
excluding Right_Left outright would have discarded the most reliable hard negatives in the corpus.

    * Right_Left PRESENT  -> EXCLUDED (45% wrong)
    * Right_Left ABSENT   -> kept, full weight (100% agreement; these are the reflection negatives)
    * Right_Top  ABSENT   -> kept at REDUCED WEIGHT (55%) rather than dropped: IR is the largest
                             deployment camera and dropping these leaves the model no IR negatives.
    * everything else     -> kept, full weight
Cell sizes are 8-36 clips, so trust the ordering more than the exact percentages.

--------------------------------------------------------------------------------------------------
SOFT TARGETS. 31% of clips have a split vote, and human agreement tracks the margin (0.726 unanimous
/ 0.864 at 4-of-5 / 0.426 at <=3/5). So the target is the normalised 5-vote distribution, not the
argmax: a 3-2 clip teaches uncertainty instead of false confidence and no rows are discarded. Train
with KL divergence.

SPLIT BY SOURCE VIDEO, never by clip -- this project shipped a clip-level leak once (an apparent
0.49 -> 0.70 gain evaporated under a video-level holdout). The TEST videos double as the reserved
pool for a future BLIND human round, so teacher-reproduction and human-accuracy land on the same
holdout and are directly comparable.

The 154 existing human-labelled behaviour clips are held out of train/val as `human_secondary`. They
are a SUPPORTING figure only, with two caveats that must travel with any number computed on them:
(1) all were labelled `assisted` (the model's answer was on screen) so they measure AGREEMENT, not
accuracy; (2) they span 65 of 82 videos, so their video-mates are in training -- clip-level overlap.

WHY A SEQUENCE, NOT A POOLED VECTOR. The previous behaviour classifier pooled CLIP features and
collapsed onto the majority classes (per-class F1 ~0). Pooling destroys time, and every class here
except Resting is defined by motion. Features are cached as [10, 512] and the model must consume the
sequence.

Output: src/dataset_etho/<version>/{manifest.jsonl, features.npz, human_secondary.jsonl, snapshot.json}
Usage:  venv/bin/python3 src/build_ethogram_dataset.py --version v1
"""
import argparse, collections, json, os, random, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import caption_openrouter as C
from ensemble_235b import extract_frames_at, interleaved_draw, DENSE_FPS, N_DRAW

VOTED = REPO / "data" / "ensemble_235b_voted.json"
INDEX = REPO / "src" / "octopus_clips_verified.json"
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]
OUTROOT = REPO / "src" / "dataset_etho"
HUMAN = [("data/human_behaviour_labels.json", "data/human_eval_sample_v1.json"),
         ("data/human_behaviour_labels_v2.json", "data/human_eval_sample_v2.json")]

ABSENT = "No octopus"
MERGE = {"Crawling": "Locomotion (crawl/swim)", "Swimming / jetting": "Locomotion (crawl/swim)"}
DROP_LABELS = {"Colour change / defensive"}
IR_ABSENT_WEIGHT = 0.5      # Right_Top absent labels agree with the human only 55% of the time
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_FRAMES = 37             # >=~15s of real footage (20s at DENSE_FPS 2.5 gives ~50 frames).
#                             A full-corpus ffprobe sweep found 147/6945 clips (2.1%) truncated to
#                             under 15s -- byte-range extraction failures that extract_clip accepted
#                             because it validates FILE SIZE, and one of them is 3.6 MB but 0.49s.
#                             8 had reached this dataset, 4 carrying a BEHAVIOUR label and one of
#                             those in TEST, where a 0.4s clip labelled "Exploration / manipulation"
#                             would have scored as a real model failure. The corpus is cleanly
#                             bimodal -- 4665 clips have 46+ frames, the 8 bad ones have <37, nothing
#                             sits in between -- so this threshold drops exactly the truncated files.
SEED = 20260822


def vid_of(clip):
    return "/".join(clip.split("/")[:2])


def merged(label):
    return MERGE.get(label, label)


def motion_features(frame_paths, pick, pix_thresh=25):
    """Per-timestep MOTION, aligned to the sampled CLIP frames. Two channels:

      inst : changed-pixel fraction vs the frame 0.4 s earlier -> instantaneous movement
      disp : changed-pixel fraction vs the previous SAMPLED frame (2 s) -> displacement over the gap

    CLIP cannot supply this. It is trained on static image-text pairs, so its embedding encodes
    appearance: an octopus crawling slowly produces nearly IDENTICAL embeddings 2 s apart because the
    content ("an octopus on sand") has not changed, while an infrared lamp flicker can move the
    embedding more than the animal does. Every class here except Resting is defined by HOW the animal
    moves, so appearance-change is the wrong signal. This is the same absolute changed-pixel measure
    the extraction gate uses (motion_detector.scan_motion_area), including its timestamp mask, so it
    is a physical measurement rather than a learned proxy.

    Computed on the already-decoded JPEGs, so it costs cv2 ops only -- no extra ffmpeg pass.
    """
    grey = {}

    def g(i):
        if i not in grey:
            im = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
            if im is None:
                return None
            h, w = im.shape
            im = im.copy()
            im[int(h * 0.88):, int(w * 0.60):] = 0        # mask the burned-in datetime
            grey[i] = im.astype(np.float32)
        return grey[i]

    def frac(a, b):
        A, B = g(a), g(b)
        if A is None or B is None or A.shape != B.shape:
            return 0.0
        return float((np.abs(A - B) > pix_thresh).mean())

    out = np.zeros((len(pick), 2), np.float32)
    for j, i in enumerate(pick):
        out[j, 0] = frac(i, i - 1) if i - 1 >= 0 else 0.0
        out[j, 1] = frac(i, pick[j - 1]) if j > 0 else 0.0
    return out


def resolve(clip):
    for r in ROOTS:
        p = r / clip
        if p.exists():
            return p
    return None


def load_human():
    """clip -> human record, for the caveated secondary eval and to hold those clips out of train."""
    out = {}
    for lf, sf in HUMAN:
        lp, sp = REPO / lf, REPO / sf
        if not (lp.exists() and sp.exists()):
            continue
        lab = json.load(open(lp))
        samp = {c["clip"]: c for c in json.load(open(sp))["clips"]}
        for k, v in lab.items():
            if k in samp and not v.get("skipped"):
                out[k] = {**v, "camera": samp[k].get("camera")}
    return out


def video_split(by_video, rng):
    """Whole videos to splits, greedy so every split carries every class. Video-level: no clip leaks."""
    dom = {v: collections.Counter(ls).most_common(1)[0][0] for v, ls in by_video.items()}
    rarity = collections.Counter(dom.values())
    order = sorted(by_video, key=lambda v: (rarity[dom[v]], rng.random()))
    counts = {s: collections.Counter() for s in SPLIT_FRACS}
    assign = {}
    for v in order:
        c = dom[v]
        best, best_def = None, None
        for s, frac in SPLIT_FRACS.items():
            tot = sum(counts[x][c] for x in SPLIT_FRACS) or 1
            deficit = frac - (counts[s][c] / tot)
            if best_def is None or deficit > best_def:
                best, best_def = s, deficit
        assign[v] = best
        for l in by_video[v]:
            counts[best][l] += 1
    return assign


def assign_splits_globally(manifest, human, mpath, rng):
    """Recompute the train/val/test assignment over the WHOLE manifest and rewrite it.

    THE BUG THIS FIXES (found by validate_ethogram_dataset.py, 2026-08-22). Features are resumable
    per clip, and manifest.jsonl is append-only, so a resumed run appends rows for the new clips and
    keeps the old ones. But the split is not a per-clip property -- video_split() is a GLOBAL greedy
    decision over the whole clip set. Run 1 saw 2,978 clips, run 2 saw 4,673, so the greedy order
    differed and 29 of the 58 videos present in both runs were assigned to a DIFFERENT split. The
    combined manifest then had 29 source videos spanning two of train/val/test, i.e. exactly the
    video-level leak the split exists to prevent. Each run was internally clean; only the
    concatenation was broken, which is why nothing errored.

    The rule: resume the EXPENSIVE per-clip work (CLIP features, minutes each), never a GLOBAL
    decision. Splits are recomputed from scratch on every run and the manifest is rewritten, which
    costs milliseconds and is leak-free by construction rather than by luck.

    Rewritten atomically (tmp + os.replace) so an interrupt cannot leave a half-written manifest --
    the file is the resume state, and a truncated one would silently shrink the dataset.
    """
    # PRUNE retroactively. The resume path skips any clip already in the manifest, so a guard added
    # in the feature loop can never remove rows written by an earlier run. Enforce it here too, where
    # every row is visible, or the 8 truncated clips that predate MIN_FRAMES would live forever.
    keep, pruned = [], []
    for r in manifest:
        (pruned if (r.get("n_frames_available") or 0) < MIN_FRAMES else keep).append(r)
    if pruned:
        for r in pruned:
            fp = mpath.parent / "feats" / (r["clip"].replace("/", "__") + ".npy")
            if fp.exists():
                fp.unlink()
        print(f"pruned {len(pruned)} truncated clips already in the manifest "
              f"(<{MIN_FRAMES} frames): " + ", ".join(sorted(r["split"] for r in pruned)))
        manifest = keep

    by_video = collections.defaultdict(list)
    for r in manifest:
        if r["clip"] not in human:          # human clips are held out regardless; they do not vote
            by_video[r["video"]].append(r["label"])
    assign = video_split(by_video, rng)
    moved = 0
    for r in manifest:
        want = "human_secondary" if r["clip"] in human else assign.get(r["video"], r["split"])
        if r["split"] != want:
            r["split"] = want; moved += 1
    tmp = Path(str(mpath) + ".tmp")
    with open(tmp, "w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, mpath)
    vs = collections.defaultdict(set)
    for r in manifest:
        vs[r["video"]].add(r["split"])
    leaks = sum(1 for s in vs.values() if len({"train", "val", "test"} & s) > 1)
    print(f"splits recomputed over all {len(manifest)} rows ({moved} reassigned); "
          f"videos spanning >1 trainable split: {leaks}")
    assert leaks == 0, f"video-level leak survived the global reassignment ({leaks} videos)"
    return manifest


def check_vote_fresh(strict=True):
    """Refuse to build from a vote file that lags the completed ensemble passes.

    This exists because it already happened. The 5-pass ensemble is resumable and ran for days;
    ensemble_235b_vote.py was run ONCE early, at ~3,444 clips, and never re-derived as the ensemble
    reached ~5,222. Every consumer reads the vote file, so the dataset was silently built from 64%
    of the labels we had already paid the 235B API for -- 1,568 fully-processed clips unused, the
    losses concentrated in Right_Top/Front/Back, the three real den angles.

    Nothing errored. The build printed a plausible clip count and ran to completion. That is the
    danger: a DERIVED artifact of a long resumable job goes stale the moment the job advances, and
    staleness looks exactly like a smaller dataset. Cheap to check, invisible if you don't.
    """
    pd = REPO / "data" / "ensemble_235b"
    if not pd.exists() or not VOTED.exists():
        return
    passes = {}
    for p in range(1, 6):
        f = pd / f"pass{p}.jsonl"
        if not f.exists():
            return
        passes[p] = {json.loads(l)["key"] for l in f.open() if l.strip()}
    complete = {k for k in set().union(*passes.values()) if all(k in passes[p] for p in passes)}
    voted = set(json.load(open(VOTED)))
    missed = complete - voted
    if not missed:
        print(f"vote is current: {len(complete)} clips with all 5 passes, all voted")
        return
    msg = (f"STALE VOTE: {len(missed)} clips have all 5 ensemble passes but are absent from\n"
           f"  {VOTED.name} ({len(voted)} voted vs {len(complete)} complete).\n"
           f"  These labels are already paid for. Re-derive before building:\n"
           f"    venv/bin/python3 src/ensemble_235b_vote.py --min-passes 5")
    if strict:
        sys.exit("\n" + msg + "\n  (--allow-stale-vote to override deliberately)")
    print("WARNING " + msg)


def main():
    """Build the frozen ethogram dataset. See check_vote_fresh for why it runs first."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-stale-vote", action="store_true",
                    help="proceed even if the vote file lags the completed ensemble passes")
    a = ap.parse_args()
    out = OUTROOT / a.version
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    check_vote_fresh(strict=not a.allow_stale_vote)
    voted = json.load(open(VOTED))
    idx = json.load(open(INDEX))
    idx = idx if isinstance(idx, list) else idx.get("clips", [])
    motion = {"/".join(str(e.get("clip_path", "")).split("/")[-3:]): e.get("mean_motion")
              for e in idx if isinstance(e, dict)}
    human = load_human()
    print(f"human labels available: {len(human)}")

    rows, drop = [], collections.Counter()
    for k, v in voted.items():
        if v.get("n_passes") != 5 or not v.get("voted"):
            drop["not_5_passes"] += 1; continue
        cam = v.get("camera")
        absent = (not v["present"]) or "not present" in str(v.get("ethogram")).lower()
        if absent:
            # directional trust: Right_Left absent is the most reliable label in the corpus (36/36)
            w = IR_ABSENT_WEIGHT if cam == "Right_Top" else 1.0
            label = ABSENT
        else:
            if cam == "Right_Left":
                drop["right_left_present_45pct_FP"] += 1; continue
            if v.get("ethogram") in DROP_LABELS:
                drop["dropped_label"] += 1; continue
            label, w = merged(v["ethogram"]), 1.0
        if resolve(k) is None:
            drop["file_missing"] += 1; continue
        rows.append((k, v, label, w))
    print(f"selected {len(rows)} clips   dropped: {dict(drop)}")
    if a.limit:
        rows = rows[:a.limit]

    classes = [ABSENT] + sorted({l for _, _, l, _ in rows if l != ABSENT})
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"classes ({len(classes)}): {classes}")

    # split by video, using only non-human rows to decide (human clips are held out either way)
    by_video = collections.defaultdict(list)
    for k, _, l, _ in rows:
        by_video[vid_of(k)].append(l)
    assign = video_split(by_video, rng)

    # RESUMABLE: features go to one .npy per clip and the manifest is appended line by line, so an
    # interruption costs a single clip. The previous version held everything in memory and wrote only
    # at the very end -- it was killed at 250/2978 and produced nothing.
    fdir = out / "feats"; fdir.mkdir(exist_ok=True)
    mpath = out / "manifest.jsonl"
    done = set()
    if mpath.exists():
        for line in open(mpath):
            line = line.strip()
            if line:
                try: done.add(json.loads(line)["clip"])
                except Exception: pass
        print(f"resuming: {len(done)} clips already featurised")

    det = C.load_detector()
    cm, pre, clf, vis, dev = det
    manifest, secondary = [], []
    mf = open(mpath, "a")
    for n, (k, v, label, w) in enumerate(rows, 1):
        if k in done:
            continue
        soft = np.zeros(len(classes), np.float32)
        if label == ABSENT:
            # presence votes give the soft target for the absent class
            top, tot = (v.get("present_votes") or "0/0").split("/")
            tot = max(1, int(tot)); nab = int(top) if not v["present"] else tot - int(top)
            soft[cidx[ABSENT]] = nab / tot
            rest = 1.0 - soft[cidx[ABSENT]]
            if rest > 0:                       # spread the residual over the behaviour votes
                dist = {merged(x): c for x, c in (v.get("all_ethograms") or {}).items()
                        if merged(x) in cidx and "not present" not in x.lower()}
                s = sum(dist.values())
                for c2, c2n in dist.items():
                    soft[cidx[c2]] += rest * c2n / s if s else 0.0
        else:
            for lab, cnt in (v.get("all_ethograms") or {}).items():
                m = merged(lab)
                if m in cidx and "not present" not in lab.lower():
                    soft[cidx[m]] += cnt
        if soft.sum() <= 0:
            drop["no_votes_after_merge"] += 1; continue
        soft /= soft.sum()

        with tempfile.TemporaryDirectory() as td:
            fr = extract_frames_at(resolve(k), td, DENSE_FPS)
            if not fr:
                drop["no_frames"] += 1; continue
            # TRUNCATED-CLIP GUARD. 4 clips yielded 1 frame instead of 10: the source mp4s are
            # 0.25-0.49 s long, not 20 s. They are byte-range extraction failures that extract_clip
            # accepted because it validates FILE SIZE (>10 KB) and a 213 KB file with a valid header
            # passes -- the same shape as the pcm_alaw bug, one layer down. A 1-frame sample has no
            # motion signal at all (both channels are 0), so it is not a degraded example, it is a
            # different kind of object; one was in the TEST split, where it would have silently
            # scored as a real failure.
            if len(fr) < MIN_FRAMES:
                drop["truncated_clip"] += 1
                print(f"  DROP truncated: {k} ({len(fr)} frames, expected ~{int(20*DENSE_FPS)})")
                continue
            pick = interleaved_draw(len(fr), N_DRAW, 1, 5)     # pass-1 grid: deterministic
            batch = torch.stack([pre(C.letterbox(Image.open(fr[i]).convert("RGB"))) for i in pick])
            with torch.no_grad():
                f = cm.encode_image(batch.to(dev)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            appearance = f.cpu().numpy().astype(np.float32)     # [10, 512] APPEARANCE sequence
            mot = motion_features(fr, pick)                     # [10, 2]   MOTION sequence
            np.save(fdir / (k.replace("/", "__") + ".npy"),
                    np.concatenate([appearance, mot], axis=1))  # [10, 514]

        split = "human_secondary" if k in human else assign[vid_of(k)]
        rec = {"clip": k, "video": vid_of(k), "split": split,
               "label": label, "label_idx": cidx[label],
               "soft": [round(float(x), 4) for x in soft], "weight": w,
               "margin": v.get("ethogram_margin"), "present_votes": v.get("present_votes"),
               "unanimous_after_merge": bool(float(soft.max()) == 1.0),
               "camera": v.get("camera"), "date": v.get("date"),
               "mean_motion": motion.get(k), "n_frames_available": len(fr), "frames_used": pick,
               "motion_inst_mean": round(float(mot[:, 0].mean()), 5),
               "motion_disp_mean": round(float(mot[:, 1].mean()), 5),
               "motion_inst_std": round(float(mot[:, 0].std()), 5),
               "feat_dim": 514, "feat_layout": "0:512 CLIP appearance | 512 motion_inst | 513 motion_disp"}
        manifest.append(rec)
        mf.write(json.dumps(rec) + "\n"); mf.flush()
        if k in human:
            h = human[k]
            secondary.append({**rec, "human_present": h.get("present"),
                              "human_ethogram": merged(h.get("ethogram")) if h.get("ethogram") else None,
                              "human_label": (ABSENT if h.get("present") is False
                                              else (merged(h["ethogram"]) if h.get("ethogram") else None)),
                              "human_assisted": h.get("assisted"), "human_seconds": h.get("seconds")})
        if n % 250 == 0:
            print(f"  featurised {n}/{len(rows)}", flush=True)

    mf.close()
    # re-read the manifest so a resumed run consolidates everything, not just this session's rows
    manifest = [json.loads(l) for l in open(mpath) if l.strip()]
    manifest = assign_splits_globally(manifest, human, mpath, rng)
    # Rebuild the human-secondary file from the FULL manifest, not this session's `secondary` list.
    # Opening it "w" with only the current session's rows silently truncated away every human clip
    # that had been featurised in an earlier run -- the same append-vs-global bug as the splits.
    secondary = [{**r, "human_present": human[r["clip"]].get("present"),
                  "human_ethogram": (merged(human[r["clip"]]["ethogram"])
                                     if human[r["clip"]].get("ethogram") else None),
                  "human_label": (ABSENT if human[r["clip"]].get("present") is False
                                  else (merged(human[r["clip"]]["ethogram"])
                                        if human[r["clip"]].get("ethogram") else None)),
                  "human_assisted": human[r["clip"]].get("assisted"),
                  "human_seconds": human[r["clip"]].get("seconds")}
                 for r in manifest if r["clip"] in human]
    with open(out / "human_secondary.jsonl", "w") as fh:
        for r in secondary:
            fh.write(json.dumps(r) + "\n")
    print(f"human_secondary rows: {len(secondary)}")
    feats = {}
    for r in manifest:
        fp = fdir / (r["clip"].replace("/", "__") + ".npy")
        if fp.exists():
            feats[r["clip"]] = np.load(fp)
    np.savez_compressed(out / "features.npz", **feats)
    print(f"consolidated {len(feats)} feature arrays -> features.npz")

    trainable = [r for r in manifest if r["split"] in SPLIT_FRACS]
    maj = (max(collections.Counter(r["label"] for r in trainable).values()) / len(trainable)) if trainable else 0
    print(f"\n{'split':<17}{'clips':>7}{'videos':>8}   " + "".join(f"{c[:13]:>15}" for c in classes))
    snap = {"version": a.version, "seed": SEED, "classes": classes, "merge": MERGE,
            "dropped_labels": sorted(DROP_LABELS),
            "camera_directional_filter": {
                "right_left_present": "EXCLUDED (45% presence FP vs human)",
                "right_left_absent": "kept, weight 1.0 (36/36 human agreement)",
                "right_top_absent": f"kept, weight {IR_ABSENT_WEIGHT} (11/20 = 55% agreement)",
                "note": "cells are 8-36 clips; trust the ordering, not the exact rates"},
            "frame_grid": {"dense_fps": DENSE_FPS, "n_frames": N_DRAW, "pass": 1},
            "soft_targets": "normalised 5-vote distribution after merge; train with KL divergence",
            "n_clips": len(manifest), "n_videos": len({r["video"] for r in manifest}),
            "majority_baseline_trainable": round(maj, 4), "drops": dict(drop), "splits": {},
            "human_secondary_caveats": [
                "all human labels were ASSISTED -> they measure AGREEMENT, not accuracy",
                "their video-mates are in train (65 of 82 videos) -> clip-level overlap",
                "SUPPORTING figure only; the primary test is held-out videos"],
            "test_videos_reserved_for_blind_human_round": []}
    for s in list(SPLIT_FRACS) + ["human_secondary"]:
        g = [r for r in manifest if r["split"] == s]
        cc = collections.Counter(r["label"] for r in g)
        vc = {c: len({r["video"] for r in g if r["label"] == c}) for c in classes}
        print(f"{s:<17}{len(g):>7}{len({r['video'] for r in g}):>8}   " +
              "".join(f"{str(cc[c])+'/'+str(vc[c])+'v':>15}" for c in classes))
        snap["splits"][s] = {"clips": len(g), "videos": sorted({r["video"] for r in g}),
                             "per_class_clips": dict(cc), "per_class_videos": vc}
    snap["test_videos_reserved_for_blind_human_round"] = snap["splits"]["test"]["videos"]
    json.dump(snap, open(out / "snapshot.json", "w"), indent=1)
    print(f"\nmajority baseline (trainable splits): {maj:.1%}   cells are clips/videos")
    print(f"test videos reserved for a future BLIND human round: {len(snap['splits']['test']['videos'])}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
