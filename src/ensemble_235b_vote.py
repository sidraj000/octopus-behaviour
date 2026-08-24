"""ensemble_235b_vote.py — majority-vote the N frame-draw passes into one label set.

Reads `data/ensemble_235b/pass*.jsonl` (whatever exists — safe to run mid-run) and writes
`data/ensemble_235b_voted.json`: one record per clip with the voted `present` and `ethogram`, the
vote counts, and the margin. Pass 1's caption is canonical; the others are kept for reference.

WHAT THE AGREEMENT NUMBERS DO AND DO NOT MEAN
  Unanimity is only evidence when the passes actually SAW different frames. Two cases where it is
  not, and which are therefore reported separately rather than pooled:
    * `deterministic` — no frame reached PRESENT_MIN, so the clip was labelled absent with no API
      call, identically every pass.
    * `sampling_varies == False` — the clip had <= N_DRAW candidate frames, so the stratified draw
      returned the same indices in every pass.
  Pooling those into an agreement statistic would inflate it. The headline agreement here is over
  clips with genuine sampling variation only.

  And the deeper caveat, from R15: voting removes VARIANCE, not BIAS. Higher agreement after
  ensembling is not evidence of higher accuracy — a systematically wrong label (e.g. the
  Exploration/manipulation sink) becomes MORE confident, not less wrong. Accuracy still needs human
  labels.

Usage
  venv/bin/python3 src/ensemble_235b_vote.py
  venv/bin/python3 src/ensemble_235b_vote.py --min-passes 3
"""
import argparse, collections, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "data" / "ensemble_235b"
OUT = REPO / "data" / "ensemble_235b_voted.json"


def read_jsonl(p):
    out = {}
    if not p.exists():
        return out
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("key"):
            out[r["key"]] = r
    return out


def majority(counter, prefer=None):
    """(winner, count, tied). Always returns a winner -- the clip is never dropped.

    `Counter.most_common(1)` breaks ties by INSERTION ORDER, i.e. whichever pass happened to be read
    from the file first, which makes the verdict depend on file layout. With 5 votes a tie is
    impossible, but abstentions and attempt-capped cells leave even vote counts where 2-2 happens.
    So a tie is broken DETERMINISTICALLY -- prefer `prefer` (pass 1's answer, the canonical one) if
    it is among the tied candidates, else the first when sorted -- and flagged via `tied` so a
    consumer can filter on confidence instead of the label silently depending on read order.
    """
    if not counter:
        return None, 0, False
    mc = counter.most_common()
    top = mc[0][1]
    cands = sorted([v for v, n in mc if n == top], key=lambda x: str(x))
    tied = len(cands) > 1
    if tied and prefer in cands:
        return prefer, top, True
    return cands[0], top, tied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-passes", type=int, default=3, help="clips with fewer votes are reported but not voted")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    files = sorted(DIR.glob("pass*.jsonl"))
    if not files:
        sys.exit(f"no pass*.jsonl in {DIR}")
    passes = {int(f.stem.replace("pass", "")): read_jsonl(f) for f in files}
    print(f"passes found: {sorted(passes)}  sizes: {{{', '.join(f'{k}:{len(v)}' for k,v in sorted(passes.items()))}}}")

    keys = set().union(*[set(v) for v in passes.values()])
    voted, stats = {}, collections.Counter()
    for k in sorted(keys):
        recs = [passes[p][k] for p in sorted(passes) if k in passes[p]]
        n_all = len(recs)
        det = any(r.get("deterministic") for r in recs)
        varies = all(r.get("sampling_varies", True) for r in recs) and not det

        # `uncertain` is an ABSTENTION, discarded from both votes. It previously counted as
        # PRESENT, because the per-pass boolean is derived as ("not present" not in label) and
        # "uncertain" does not contain that string -- so a pass that admitted it could not tell was
        # silently tallied as "the animal is there". 230 of 5,671 passes answered uncertain.
        eff = [r for r in recs if str(r.get("ethogram", "")).strip().lower() != "uncertain"]
        n = len(eff)
        n_abstain = n_all - n

        p1 = (passes[min(passes)].get(k) or {})
        eth = collections.Counter(r["ethogram"] for r in eff)
        pres = collections.Counter(bool(r["present"]) for r in eff)
        top_e, top_e_n, tie_e = majority(eth, prefer=p1.get("ethogram"))
        top_p, top_p_n, tie_p = majority(pres, prefer=p1.get("present"))
        rec = {
            "n_votes": n, "n_passes": n_all, "n_abstain": n_abstain,
            "deterministic": det, "sampling_varies": varies,
            "camera": recs[0].get("camera"), "date": recs[0].get("date"),
            "frames_available": recs[0].get("frames_available"),
            "present": top_p, "present_votes": f"{top_p_n}/{n}" if n else "0/0",
            "present_tied": tie_p,
            "ethogram": top_e, "ethogram_votes": f"{top_e_n}/{n}" if n else "0/0",
            "ethogram_tied": tie_e,
            "ethogram_margin": round(top_e_n / n, 3) if n else None,
            "unanimous": bool(n) and top_e_n == n,
            "all_ethograms": dict(eth),
            "caption": p1.get("caption"),
            # Every clip with at least one non-abstaining vote KEEPS a verdict. Low vote counts and
            # ties are recorded (n_votes / margin / *_tied) so consumers filter on confidence --
            # they are never a reason to drop the clip. Only an all-uncertain clip has no verdict.
            "voted": n >= 1,
            "low_confidence": n < a.min_passes or tie_e or tie_p,
            "no_verdict": n == 0,
        }
        voted[k] = rec
        stats["clips"] += 1
        stats["deterministic" if det else ("varies" if varies else "no_variation")] += 1
        if rec["no_verdict"]:
            stats["no_verdict_all_uncertain"] += 1
        if rec["ethogram_tied"] or rec["present_tied"]:
            stats["tied"] += 1
        if n_abstain:
            stats["had_abstentions"] += 1
        if rec["voted"] and varies:
            stats["unanimous" if rec["unanimous"] else "split"] += 1

    print(f"\nclips: {stats['clips']}")
    print(f"  deterministic-absent (no API, identical every pass): {stats['deterministic']}")
    print(f"  no sampling variation (<= N_DRAW frames)           : {stats['no_variation']}")
    print(f"  genuine sampling variation                         : {stats['varies']}")
    tot = stats["unanimous"] + stats["split"]
    if tot:
        print(f"\namong the {tot} with variation AND >= {a.min_passes} votes:")
        print(f"  unanimous on ethogram : {stats['unanimous']} ({stats['unanimous']/tot:.1%})")
        print(f"  split                 : {stats['split']} ({stats['split']/tot:.1%})")
    # distribution of the voted label, and how often the vote overturns a single pass
    d = collections.Counter(v["ethogram"] for v in voted.values() if v["voted"])
    print("\nvoted ethogram distribution:")
    for k2, n2 in d.most_common():
        print(f"  {str(k2)[:38]:<38} {n2:>5}  {n2/max(1,sum(d.values())):5.1%}")
    p1 = passes[min(passes)]
    flip = sum(1 for k2, v in voted.items()
               if v["voted"] and k2 in p1 and p1[k2]["ethogram"] != v["ethogram"])
    print(f"\nvote differs from pass-1 label on {flip} clips "
          f"({flip/max(1,sum(1 for v in voted.values() if v['voted'])):.1%})")
    json.dump(voted, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
