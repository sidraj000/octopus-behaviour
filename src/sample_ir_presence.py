"""sample_ir_presence.py — draw a blind labelling round aimed at the IR presence labels.

WHY THIS ROUND EXISTS
The 6-class ethogram classifier's dominant error is `No octopus` <-> `Resting / stationary`: 22.3%
of all test errors (R28). Weighting fixes did not touch it, and they should not have -- it is a LABEL
problem. Measured against the 298 existing human labels:

    camera   ensemble says "No octopus"    human agrees it is empty
    IR (Right_Top)   n=20                  11/20 = 55%   <- the other 45% were ALL "Resting"
    colour           n=75                  72/75 = 96%

So roughly half the IR clips the ensemble calls empty contain a resting animal, and the classifier is
being penalised for predicting `Resting` correctly on them. `No octopus` is one of the six ethogram
classes (not a presence pre-filter), so these labels ARE ethogram labels and fixing them is an
ethogram improvement.

The blocker is sample size: 11/20 puts the 45% error rate somewhere in a 95% CI of about 32-77%,
which is far too wide to correct 521 clips or to set a soft target from.

DESIGN -- three groups, deliberately mixed
    ir_absent   (target)  IR clips the ensemble called `No octopus`
    ir_present  (reverse) IR clips the ensemble called present -- the other error direction
    colour_absent (control) colour clips called `No octopus`, where we expect ~96% agreement

The mix is methodological, not decorative. A round consisting only of clips the ensemble called
empty would shift the labeller's prior toward "empty" and the resulting rate would measure that
drift rather than the labels. The control group also detects a broken round: if colour_absent does
NOT come back near 96%, something is wrong with the pipeline or the UI, not with IR.

Group identity lives ONLY in `_group`, which shares the `_`-prefix convention the UI already
withholds from the labeller, so the round stays blind.

Spread across source videos (MAX_PER_VIDEO=2) because the video count is the real sample size, and
excludes every clip already labelled in earlier rounds so no effort is repeated.

Usage: venv/bin/python3 src/sample_ir_presence.py            # writes data/human_eval_sample_v3.json
       EVAL_VERSION=v3 venv/bin/python3 ui/label_ethogram.py # label it
"""
import argparse, collections, json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "src" / "dataset_etho" / "v1" / "manifest.jsonl"
VOTED = REPO / "data" / "ensemble_235b_voted.json"
PRIOR_LABELS = ["data/human_behaviour_labels.json", "data/human_behaviour_labels_v2.json"]
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]

IR = "Right_Top"
ABSENT = "No octopus"
SEED = 20260822
MAX_PER_VIDEO = 2
GROUPS = {"ir_absent": 120, "ir_present": 40, "colour_absent": 40}


def on_disk(clip):
    return any((r / clip).exists() and (r / clip).stat().st_size > 10000 for r in ROOTS)


def already_labelled():
    done = set()
    for p in PRIOR_LABELS:
        f = REPO / p
        if f.exists():
            d = json.load(open(f))
            done |= set(d if isinstance(d, list) else d.keys())
    return done


def fill(pool, n, rng):
    """Take up to n clips, at most MAX_PER_VIDEO per source video, video-diverse first.

    Round-robin over videos rather than a flat sample: a flat draw on this corpus concentrates on
    the few videos with many clips, and the video count is the real sample size.
    """
    by_vid = collections.defaultdict(list)
    for c in pool:
        by_vid[c["source_video"]].append(c)
    for v in by_vid:
        rng.shuffle(by_vid[v])
    vids = sorted(by_vid)
    rng.shuffle(vids)
    out = []
    for k in range(MAX_PER_VIDEO):
        for v in vids:
            if len(out) >= n:
                return out
            if len(by_vid[v]) > k:
                out.append(by_vid[v][k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "human_eval_sample_v3.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = random.Random(SEED)

    man = [json.loads(l) for l in open(DATASET) if l.strip()]
    voted = json.load(open(VOTED))
    done = already_labelled()
    print(f"dataset clips: {len(man)}   already human-labelled: {len(done)}")

    cand = collections.defaultdict(list)
    skipped = collections.Counter()
    for r in man:
        k = r["clip"]
        if k in done:
            skipped["already_labelled"] += 1; continue
        if not on_disk(k):
            skipped["not_on_disk"] += 1; continue
        v = voted.get(k) or {}
        rec = {"clip": k, "source_video": r["video"], "camera": r["camera"], "date": r.get("date"),
               "_model_present": v.get("present"), "_model_ethogram": v.get("ethogram"),
               "_model_votes": v.get("ethogram_votes"), "_model_margin": v.get("ethogram_margin"),
               "_model_unanimous": v.get("unanimous"), "_model_low_conf": v.get("low_confidence")}
        ir, absent = r["camera"] == IR, r["label"] == ABSENT
        if ir and absent:
            cand["ir_absent"].append({**rec, "_group": "ir_absent"})
        elif ir and not absent:
            cand["ir_present"].append({**rec, "_group": "ir_present"})
        elif absent:
            cand["colour_absent"].append({**rec, "_group": "colour_absent"})
        else:
            skipped["colour_present_not_in_scope"] += 1
    print("skipped:", dict(skipped))
    print("candidate pool per group:", {g: len(v) for g, v in cand.items()})

    clips, short = [], {}
    for g, want in GROUPS.items():
        got = fill(cand[g], want, rng)
        clips.extend(got)
        if len(got) < want:
            short[g] = (len(got), want)
        print(f"  {g:<15} {len(got):>4}/{want:<4} from {len({c['source_video'] for c in got})} videos")
    if short:
        print(f"  NOTE under-filled (pool exhausted, not silently topped up from another group): {short}")
    rng.shuffle(clips)                      # interleave groups so the order carries no signal

    out = {"description": "frozen human ethogram-labelling sample (round 3, IR presence). Fields "
                          "prefixed _model_ are the ensemble verdict and _group is the sampling "
                          "stratum; NEITHER may be shown to the labeller.",
           "purpose": "measure the IR No-octopus label error rate (prior estimate 45% on n=20) with "
                      "a reverse-direction arm and a colour control",
           "seed": SEED, "max_per_video": MAX_PER_VIDEO, "n": len(clips),
           "n_source_videos": len({c["source_video"] for c in clips}),
           "groups": {g: sum(1 for c in clips if c["_group"] == g) for g in GROUPS},
           "by_camera": dict(collections.Counter(c["camera"] for c in clips)),
           "expected": {"colour_absent": "~96% agree-empty (control; a big miss means the round or "
                                         "the UI is broken, not that colour labels are bad)",
                        "ir_absent": "prior 55% agree-empty -- this is the number the round exists "
                                     "to pin down"},
           "clips": clips}
    if a.dry_run:
        print("\n[dry-run] nothing written."); return
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}  ({len(clips)} clips / {out['n_source_videos']} videos)")
    print(f"label with: EVAL_VERSION=v3 venv/bin/python3 ui/label_ethogram.py   -> http://localhost:8021")


if __name__ == "__main__":
    main()
