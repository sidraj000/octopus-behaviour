"""sample_human_eval.py — freeze the 100-clip set for HUMAN ethogram labelling.

WHY. R15 measured the behaviour labels' *consistency* (kappa 0.552) and the paper says outright that
"validation against a human ethologist remains the larger open item". Nothing in this project has ever
measured whether a behaviour label is CORRECT: ~900 hand labels exist for masks, presence negatives
and hard negatives, and effectively zero for behaviour. This freezes the set that closes that.

SAMPLING DESIGN (and why each part is there)
  * Only clips with ALL 5 ensemble passes, so every labelled clip has a full vote to compare against.
  * SAMPLED BY SOURCE VIDEO, capped at MAX_PER_VIDEO. Clips from one recording are near-duplicates;
    the project rule is video-level splits, never clip-level. NOTE the hard limit this implies: the
    eligible pool spans only ~56 distinct videos, so 100 clips is ~2 per video and the effective
    sample size for generalisation is the VIDEO count, not 100.
  * Strata over the voted label, deliberately over-sampling the rare and the surprising:
      - absent-voted clips get 30, because presence gates every downstream aggregate and not one
        absent verdict has ever been checked by a human.
      - Crawling and Resting are over-weighted relative to their share because the ensemble moved
        them most (2%->22% and 33%->11%); those are the shifts most in need of verification.
      - Swimming/jetting is the least self-consistent class (10% unanimous), so it gets a floor.
  * Within each stratum, HALF unanimous and HALF split where supply allows, so accuracy can be
    measured AS A FUNCTION OF VOTE MARGIN -- that is what turns the margin into a calibrated
    confidence rather than a decoration.
  * A weak-margin cell (margin <= 0.5) regardless of label, to test directly whether low margin
    predicts wrongness.

WHAT THIS SAMPLE CANNOT DO. Because rare classes and contested clips are over-sampled, the raw
pass-rate is NOT corpus accuracy. Per-class and per-margin accuracy are valid immediately; an overall
figure needs post-stratification against the final label distribution once the full run lands. Also
Right_Top (infrared) is barely in the eligible pool yet (~15 clips), so IR accuracy stays unmeasured.

Output: data/human_eval_sample_v1.json  (frozen; commit it, do not regenerate to suit a result)
"""
import argparse, collections, json, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VOTED = REPO / "data" / "ensemble_235b_voted.json"
OUT = REPO / "data" / "human_eval_sample_v1.json"

# 3, not 2: the eligible pool spans only ~56 videos, and a cap of 2 could not fill the strata (it
# yielded 61 of 100). Still video-level-ish independence; the analysis must report the VIDEO count.
MAX_PER_VIDEO = 3
MAX_PER_CAMERA_FRAC = 0.35   # no camera may exceed this share, or Right_Left/Right_Top dominate
SEED = 20260821

# stratum -> target n. Keys are voted-label values; "__weak__" is the low-margin diagnostic cell.
TARGETS = {
    "octopus not present": 30,
    "Exploration / manipulation": 12,
    "Crawling": 12,
    "Reaching out of water": 12,
    "Resting / stationary": 10,
    "Human / enrichment interaction": 8,
    "Swimming / jetting": 6,
    "Colour change / defensive": 2,
    "__weak__": 8,
}

# v2 is shaped by what v1 actually revealed (98 clips, all assisted):
#   * presence errors were ENTIRELY one-directional -- 18 model-present/human-absent, 0 the other
#     way. So weight model-PRESENT clips to pin down the false-positive rate.
#   * Resting <-> Reaching-out was the single biggest behaviour confusion (4 of 12 errors).
#   * the vote margin did NOT predict correctness (73.9% at 5/5 vs 73.7% at <=3/5) -- worth
#     re-testing with more n before treating that negative as settled.
#   * Right_Top (infrared) was unmeasurable in v1 (3 clips in the pool); there are 922 now.
TARGETS_V2 = {
    "Resting / stationary": 25,
    "Reaching out of water": 25,
    "Exploration / manipulation": 20,
    "Crawling": 20,
    "Human / enrichment interaction": 15,
    "Swimming / jetting": 14,
    "Colour change / defensive": 1,
    "octopus not present": 60,
    "__weak__": 20,
}


def source_video(key):
    return "/".join(key.split("/")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-per-video", type=int, default=MAX_PER_VIDEO)
    ap.add_argument("--targets", choices=["v1", "v2"], default="v1")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="label files whose clips are already done and must not be re-drawn")
    a = ap.parse_args()
    global TARGETS
    if a.targets == "v2":
        TARGETS = TARGETS_V2

    voted = json.load(open(VOTED))
    already = set()
    for f in a.exclude:
        try:
            already |= set(json.load(open(f)))
        except Exception:
            print(f"  (could not read {f})")
    pool = {k: v for k, v in voted.items()
            if v.get("n_passes") == 5 and v.get("voted") and k not in already}
    if already:
        print(f"excluding {len(already)} already-labelled clips")
    print(f"eligible (all 5 passes): {len(pool)} clips / "
          f"{len({source_video(k) for k in pool})} source videos")

    rng = random.Random(SEED)
    used_per_video = collections.Counter()
    used_per_camera = collections.Counter()
    chosen, by_stratum = {}, collections.Counter()
    cam_cap = None   # set once the total target is known

    def take(cands, n, stratum):
        """Pick n, alternating unanimous/split so margin is spread, respecting the per-video cap."""
        una = [k for k in cands if voted[k].get("unanimous")]
        spl = [k for k in cands if not voted[k].get("unanimous")]
        rng.shuffle(una); rng.shuffle(spl)
        out = []
        while len(out) < n and (una or spl):
            for src in ((una, spl) if len(out) % 2 == 0 else (spl, una)):
                while src:
                    k = src.pop()
                    if k in chosen:
                        continue
                    if used_per_video[source_video(k)] >= a.max_per_video:
                        continue
                    if cam_cap and used_per_camera[voted[k].get("camera")] >= cam_cap:
                        continue
                    out.append(k); used_per_video[source_video(k)] += 1
                    used_per_camera[voted[k].get("camera")] += 1
                    break
                if len(out) >= n:
                    break
        for k in out:
            chosen[k] = stratum
            by_stratum[stratum] += 1
        return len(out)

    cam_cap = int(sum(TARGETS.values()) * MAX_PER_CAMERA_FRAC)
    print(f"per-camera cap: {cam_cap} clips ({MAX_PER_CAMERA_FRAC:.0%} of {sum(TARGETS.values())})")

    # 1. the weak-margin diagnostic cell first -- it is the scarcest constraint
    weak = [k for k, v in pool.items() if (v.get("ethogram_margin") or 1) <= 0.5]
    got = take(weak, TARGETS["__weak__"], "__weak__")
    print(f"  __weak__ (margin<=0.5): {got}/{TARGETS['__weak__']} (pool {len(weak)})")

    # 2. the label strata, SCARCEST FIRST. Order matters because every pick consumes a slot from
    # that clip's source video, and the cap is the binding constraint. Filling the abundant strata
    # first (absent has 662 candidates) starves the rare classes of videos -- doing that yielded
    # 0/6 swimming and 0/8 human-interaction. Rarest-first gives the scarce classes first refusal.
    label_strata = [(lab, n) for lab, n in TARGETS.items() if lab != "__weak__"]
    supply = {lab: sum(1 for v in pool.values() if v.get("ethogram") == lab)
              for lab, _ in label_strata}
    for lab, n in sorted(label_strata, key=lambda x: supply[x[0]]):
        cands = [k for k, v in pool.items() if v.get("ethogram") == lab and k not in chosen]
        got = take(cands, n, lab)
        flag = "" if got == n else f"   <-- SHORT (pool {len(cands)})"
        print(f"  {lab[:34]:<34} {got}/{n}{flag}")

    # freeze, with the model's verdict stored for LATER comparison -- the UI must never show it
    recs = []
    for k, stratum in chosen.items():
        v = voted[k]
        recs.append({
            "clip": k, "stratum": stratum, "source_video": source_video(k),
            "camera": v.get("camera"), "date": v.get("date"),
            # ---- held back from the labeller, used only in the analysis ----
            "_model_present": v.get("present"), "_model_ethogram": v.get("ethogram"),
            "_model_votes": v.get("ethogram_votes"), "_model_margin": v.get("ethogram_margin"),
            "_model_unanimous": v.get("unanimous"), "_model_low_conf": v.get("low_confidence"),
        })
    recs.sort(key=lambda r: r["clip"])
    rng.shuffle(recs)          # present in random order so the labeller cannot infer the stratum

    out = {"description": "frozen human ethogram-labelling sample; fields prefixed _model_ are the "
                          "ensemble verdict and MUST NOT be shown to the labeller",
           "seed": SEED, "max_per_video": a.max_per_video, "n": len(recs),
           "n_source_videos": len({r["source_video"] for r in recs}),
           "by_stratum": dict(by_stratum),
           "by_camera": dict(collections.Counter(r["camera"] for r in recs)),
           "clips": recs}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\ntotal {len(recs)} clips / {out['n_source_videos']} source videos")
    print(f"  by camera: {out['by_camera']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
