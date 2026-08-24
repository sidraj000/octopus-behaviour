"""eval_ethogram_human.py — score the ethogram classifier against HUMAN labels, and the teacher too.

The ladder (R27/R28) scores the model against the 5-pass 235B ensemble, i.e. against its own teacher.
That measures TEACHER REPRODUCTION, not correctness: a model could reproduce the teacher perfectly and
still be wrong wherever the teacher is. 456 human labels exist, so the more useful question is
answerable — but only if the populations are kept apart.

THREE POPULATIONS, NEVER POOLED. Pooling them would average a contaminated set into a clean one and
produce a number that means nothing:

  train_CONTAMINATED (102 clips)  The model TRAINED on these clips (with the ensemble label). Scoring
                                  it here measures memorisation. Computed and printed only so the gap
                                  to the held-out sets is visible; never reported as performance.
  human_secondary   (251 clips)   Held out of train/val/test by construction. The honest caveat is
                                  that 97 of their 99 videos also appear in training, so this is
                                  clip-disjoint but not video-disjoint.
  test              (~30 clips)   Fully video-disjoint from training, so this is the only clean
                                  human-accuracy figure -- and it is small, so its CI is wide and is
                                  printed rather than hidden.

WHAT ELSE THIS MEASURES, and the reason it matters more than the model score: on the same clips it
scores the **TEACHER** against the human. That is the LABEL CEILING. If the ensemble is only ~70%
accurate, a student at macro-F1 0.53 may be near the maximum achievable and more teacher-labelled
data buys little; if the ensemble is ~90%, the headroom is real. Those imply opposite next steps, and
the ladder alone cannot distinguish them.

CAVEAT CARRIED ON EVERY NUMBER: all 456 human labels were recorded `assisted` -- the model's answer
was on screen -- so they measure AGREEMENT, not accuracy, and every figure here is an optimistic
bound. The UI defect that caused it is fixed (per-round hint key, default off); a genuinely blind
round on the reserved test videos is the open item this script is built to consume.

Usage: venv/bin/python3 src/eval_ethogram_human.py --version v1 [--rung 3] [--backbone clip]
"""
import argparse, collections, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import train_ethogram as T

HUMAN = ["data/human_behaviour_labels.json", "data/human_behaviour_labels_v2.json",
         "data/human_behaviour_labels_v3.json"]
MERGE = {"Crawling": "Locomotion (crawl/swim)", "Swimming / jetting": "Locomotion (crawl/swim)"}
ABSENT = "No octopus"


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def human_labels(classes):
    """clip -> merged human class, plus the flags any honest report has to split on."""
    out = {}
    for p in HUMAN:
        f = REPO / p
        if not f.exists():
            continue
        for k, v in json.load(open(f)).items():
            if v.get("skipped"):
                continue
            lab = ABSENT if v.get("present") is False else MERGE.get(v.get("ethogram"), v.get("ethogram"))
            if lab is None or lab not in classes:
                continue
            out[k] = {"label": lab, "assisted": bool(v.get("assisted")),
                      "unsure": bool(v.get("unsure")), "seconds": v.get("seconds")}
    return out


def score(pred_idx, true_idx, n_cls, classes):
    f1, per = T.macro_f1(np.asarray(pred_idx), np.asarray(true_idx), n_cls)
    acc = float((np.asarray(pred_idx) == np.asarray(true_idx)).mean())
    return {"macro_f1": round(f1, 4), "accuracy": round(acc, 4),
            "per_class": {classes[c]: per[c] for c in per}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--rung", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--out", default=str(REPO / "data" / "ethogram_human_eval.json"))
    a = ap.parse_args()

    man, X, classes, D = T.load(a.version, a.backbone)
    cidx = {c: i for i, c in enumerate(classes)}
    hum = human_labels(set(classes))
    by_clip = {r["clip"]: r for r in man}
    print(f"human labels usable: {len(hum)}   CW_POWER={T.CW_POWER}  rung={a.rung}")

    # ---- partition, and refuse to pool ----
    pops = collections.defaultdict(list)
    for k, h in hum.items():
        r = by_clip.get(k)
        if r is None:
            continue
        pop = "train_CONTAMINATED" if r["split"] == "train" else r["split"]
        pops[pop].append((k, h, r))
    print("\npopulations (never pooled):")
    for p, rows in sorted(pops.items(), key=lambda kv: -len(kv[1])):
        print(f"  {p:<20}{len(rows):>5} clips / {len({r['video'] for _,_,r in rows}):>3} videos")

    # ---- the STUDENT on the same clips: average softmax over seeds before deciding ----
    # Averaging probabilities, not argmaxes: one seed's confident error should not outvote two seeds'
    # correct uncertainty. Train rows are included in the forward pass but scored separately.
    extra = [r for _, _, r in sum(pops.values(), [])]
    acc_p, clips_order = None, None
    for s in range(a.seeds):
        out = T.run_one(a.rung, man, X, classes, seed=s, extra_rows=extra, D=D)
        p = np.asarray(out["extra_probs"], np.float32)
        acc_p = p if acc_p is None else acc_p + p
        clips_order = out["extra_clips"]
    student = {k: int(np.argmax(v)) for k, v in zip(clips_order, acc_p / a.seeds)}
    print(f"\nstudent predictions over {len(student)} human-labelled clips "
          f"({a.seeds} seeds, probabilities averaged)")

    results = {"version": a.version, "rung": a.rung, "cw_power": T.CW_POWER,
               "caveat": "ALL human labels are `assisted` (model answer visible) -> agreement, not "
                         "accuracy; every figure is an optimistic bound",
               "populations": {}}

    for pop, rows in pops.items():
        clips = [k for k, _, _ in rows]
        truth = [cidx[h["label"]] for _, h, _ in rows]
        n_vid = len({r["video"] for _, _, r in rows})
        entry = {"n_clips": len(rows), "n_videos": n_vid,
                 "class_counts": dict(collections.Counter(classes[t] for t in truth))}

        # --- the TEACHER vs the human on these same clips: the LABEL CEILING ---
        teach = [cidx[by_clip[k]["label"]] for k in clips]
        agree = int(sum(1 for t, g in zip(teach, truth) if t == g))
        lo, hi = wilson(agree, len(rows))
        entry["teacher_vs_human"] = {**score(teach, truth, len(classes), classes),
                                     "agreement": round(agree / len(rows), 4),
                                     "agreement_ci95": [round(lo, 4), round(hi, 4)]}
        # --- the STUDENT vs the human, and vs the teacher, on the SAME clips ---
        stu = [student[k] for k in clips]
        s_agree = int(sum(1 for p, g in zip(stu, truth) if p == g))
        slo, shi = wilson(s_agree, len(rows))
        entry["student_vs_human"] = {**score(stu, truth, len(classes), classes),
                                     "agreement": round(s_agree / len(rows), 4),
                                     "agreement_ci95": [round(slo, 4), round(shi, 4)]}
        entry["student_vs_teacher"] = score(stu, teach, len(classes), classes)
        # Where the student and the teacher DISAGREE, who does the human side with? This is the only
        # direct evidence on whether the student's "errors" are errors or teacher noise.
        d = [(p, t, g) for p, t, g in zip(stu, teach, truth) if p != t]
        if d:
            entry["where_they_disagree"] = {
                "n": len(d),
                "human_sides_with_student": int(sum(1 for p, _, g in d if p == g)),
                "human_sides_with_teacher": int(sum(1 for _, t, g in d if t == g)),
                "human_agrees_with_neither": int(sum(1 for p, t, g in d if g != p and g != t))}

        if pop != "train_CONTAMINATED":
            print(f"\n=== {pop}  ({len(rows)} clips / {n_vid} videos) ===")
            print(f"  TEACHER (5-pass 235B) vs human : agreement {agree}/{len(rows)} = "
                  f"{agree/len(rows):.1%}  95% CI [{lo:.1%}, {hi:.1%}]   "
                  f"macro-F1 {entry['teacher_vs_human']['macro_f1']:.4f}")
            print("     ^ this is the LABEL CEILING -- the student is trained to reproduce these")
            print(f"  STUDENT vs human               : agreement {s_agree}/{len(rows)} = "
                  f"{s_agree/len(rows):.1%}  95% CI [{slo:.1%}, {shi:.1%}]   "
                  f"macro-F1 {entry['student_vs_human']['macro_f1']:.4f}")
            print(f"  STUDENT vs teacher             : macro-F1 "
                  f"{entry['student_vs_teacher']['macro_f1']:.4f}  (reproduction, not correctness)")
            if d:
                w = entry["where_they_disagree"]
                print(f"  where student and teacher disagree (n={w['n']}): human sides with the "
                      f"STUDENT {w['human_sides_with_student']}, with the TEACHER "
                      f"{w['human_sides_with_teacher']}, neither {w['human_agrees_with_neither']}")
        results["populations"][pop] = entry

    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out}")
    print("\nREAD THIS BEFORE QUOTING ANY NUMBER ABOVE:")
    print("  * every human label was `assisted` -> agreement, not accuracy; optimistic bound")
    print("  * human_secondary is clip-disjoint but NOT video-disjoint from training")
    print("  * the test population is the only clean one and is small -- quote its CI, not its point")
    print("  * train_CONTAMINATED is diagnostic only and must never be reported as performance")


if __name__ == "__main__":
    main()
