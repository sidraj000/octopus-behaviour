"""eval_human_vs_ensemble.py — score the 235B ensemble against HUMAN labels.

Closes (partially) the gap the paper names outright: "validation against a human ethologist remains
the larger open item". R15 measured the behaviour labels' CONSISTENCY (kappa 0.552) and never their
accuracy; ~900 hand labels existed for masks/presence/hard-negatives and effectively zero for
behaviour.

READ THE `assisted` FLAG BEFORE QUOTING ANYTHING. The labelling UI can show the model's verdict on
request, and every label records whether it was visible when committed:
  * assisted=False -> the human judged independently  -> this is ACCURACY.
  * assisted=True  -> the model's answer was on screen -> this is AGREEMENT, and anchoring inflates
    it. Both rounds so far are 100% assisted, so every number below is agreement, not accuracy.
This script reports the two populations separately and never pools them.

Inputs (per round): data/human_eval_sample_{v}.json  (frozen; carries the _model_* verdict)
                    data/human_behaviour_labels[_{v}].json
Output:             data/human_vs_ensemble_results.json

Usage: venv/bin/python3 src/eval_human_vs_ensemble.py
"""
import collections, itertools, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "human_vs_ensemble_results.json"
ROUNDS = [("v1", "human_eval_sample_v1.json", "human_behaviour_labels.json"),
          ("v2", "human_eval_sample_v2.json", "human_behaviour_labels_v2.json")]


def kappa_binary(tt, tf, ft, ff):
    n = tt + tf + ft + ff
    if not n:
        return None, None
    po = (tt + ff) / n
    pa, pb = (tt + tf) / n, (tt + ft) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round(po, 4), (round((po - pe) / (1 - pe), 4) if pe < 1 else None)


def analyse(pairs):
    """pairs: list of (human_record, sample_record_with_model_verdict)."""
    hp = lambda v: v.get("present")
    mp = lambda c: c.get("_model_present")
    tt = sum(1 for v, c in pairs if hp(v) and mp(c))
    ff = sum(1 for v, c in pairs if not hp(v) and not mp(c))
    tf = sum(1 for v, c in pairs if hp(v) and not mp(c))
    ft = sum(1 for v, c in pairs if not hp(v) and mp(c))
    agr, kap = kappa_binary(tt, tf, ft, ff)
    out = {"n": len(pairs),
           "presence": {"human_present_model_present": tt, "human_present_model_absent": tf,
                        "human_absent_model_present": ft, "human_absent_model_absent": ff,
                        "agreement": agr, "kappa": kap,
                        "model_false_positives": ft,
                        "model_fp_rate_of_model_present": round(ft / (tt + ft), 4) if (tt + ft) else None,
                        "model_false_negatives": tf}}
    both = [(v, c) for v, c in pairs if hp(v) and mp(c)]
    ok = sum(1 for v, c in both if v.get("ethogram") == c.get("_model_ethogram"))
    out["behaviour"] = {"n_both_present": len(both),
                        "exact_match": ok,
                        "exact_match_rate": round(ok / len(both), 4) if both else None,
                        "confusions_human_to_model": [
                            {"human": a, "model": b, "n": k} for (a, b), k in
                            collections.Counter((v.get("ethogram"), c.get("_model_ethogram"))
                                                for v, c in both
                                                if v.get("ethogram") != c.get("_model_ethogram")).most_common(12)]}
    # does the ensemble's own vote margin predict whether the human agrees?
    bands = {}
    for lo, hi, lab in ((1.0, 1.01, "unanimous_5of5"), (0.8, 1.0, "margin_0.8"), (0.0, 0.8, "margin_le_0.6")):
        g = [(v, c) for v, c in both if lo <= (c.get("_model_margin") or 0) < hi]
        if g:
            bands[lab] = {"n": len(g),
                          "agree_rate": round(sum(1 for v, c in g
                                                  if v.get("ethogram") == c.get("_model_ethogram")) / len(g), 4)}
    out["behaviour"]["by_vote_margin"] = bands
    # presence FP by camera -- this is where the reflection camera shows up
    cam = collections.defaultdict(lambda: [0, 0])
    for v, c in pairs:
        if mp(c):
            cam[c.get("camera")][0] += 1
            if not hp(v):
                cam[c.get("camera")][1] += 1
    out["presence"]["by_camera"] = {k: {"model_present": a, "human_says_absent": b,
                                        "fp_rate": round(b / a, 4) if a else None}
                                   for k, (a, b) in sorted(cam.items(), key=lambda x: -x[1][0])}
    return out


def main():
    res = {"_meta": {"note": "assisted=True labels measure AGREEMENT (the model's answer was on "
                             "screen); assisted=False measure ACCURACY. Never pool them.",
                     "rounds": []}}
    for ver, sfile, hfile in ROUNDS:
        sp, hpth = REPO / "data" / sfile, REPO / "data" / hfile
        if not (sp.exists() and hpth.exists()):
            continue
        samp = {c["clip"]: c for c in json.load(open(sp))["clips"]}
        hum = json.load(open(hpth))
        pairs = [(v, samp[k]) for k, v in hum.items() if k in samp and not v.get("skipped")]
        blind = [(v, c) for v, c in pairs if not v.get("assisted")]
        asst = [(v, c) for v, c in pairs if v.get("assisted")]
        secs = sorted(v.get("seconds", 0) for v, _ in pairs if v.get("seconds"))
        r = {"round": ver, "n_labelled": len(hum), "n_sample": len(samp),
             "n_source_videos": len({"/".join(k.split("/")[:2]) for k in hum}),
             "n_blind": len(blind), "n_assisted": len(asst),
             "median_seconds_per_clip": secs[len(secs) // 2] if secs else None,
             "human_label_distribution": dict(collections.Counter(
                 v.get("ethogram") or ("NO OCTOPUS" if v.get("present") is False else "-")
                 for v in hum.values()).most_common()),
             "AGREEMENT_assisted": analyse(asst) if asst else None,
             "ACCURACY_blind": analyse(blind) if blind else None}
        res["_meta"]["rounds"].append(ver)
        res[ver] = r
        print(f"\n=== round {ver}: {r['n_labelled']}/{r['n_sample']} labelled, "
              f"{r['n_source_videos']} videos | blind {r['n_blind']} / assisted {r['n_assisted']} ===")
        for kind in ("ACCURACY_blind", "AGREEMENT_assisted"):
            d = r[kind]
            if not d:
                print(f"  {kind}: none"); continue
            p, b = d["presence"], d["behaviour"]
            print(f"  {kind}  n={d['n']}")
            print(f"    presence agreement {p['agreement']}  kappa {p['kappa']}  "
                  f"FP {p['model_false_positives']} ({p['model_fp_rate_of_model_present']}) "
                  f"FN {p['model_false_negatives']}")
            print(f"    behaviour exact match {b['exact_match']}/{b['n_both_present']} = {b['exact_match_rate']}")
            print(f"    by margin: " + "  ".join(f"{k}={v['agree_rate']}(n={v['n']})"
                                                 for k, v in b["by_vote_margin"].items()))
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
