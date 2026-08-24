"""Deployment test: does the segmenter's mask-area separate octopus PRESENT vs ABSENT/REFLECTION?
This is the real Phase-3 payoff — a presence gate that beats the CLIP gate's reflection FPs —
and matters even if pixel IoU < 0.85."""
import json, subprocess, tempfile, glob, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home()))
from segment_octopus import OctoSegmenter
from train_segmenter import source_video

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(Path.home()/"weights_seg/octo_seg_aug_lraspp.pt")
seg = OctoSegmenter(CKPT)
print(f"model={CKPT}  val(IoU)={seg.val}")
DS = Path.home()/"dataset_seg/v1"

# POSITIVES: val present frames (octopus definitely present)
rows = [json.loads(l) for l in open(DS/"manifest.jsonl")]
rng = np.random.RandomState(42); vids = sorted({source_video(r["clip"]) for r in rows}); rng.shuffle(vids)
val_vids = set(vids[:max(1, int(len(vids)*0.2))])
val = [r for r in rows if source_video(r["clip"]) in val_vids]
import random; random.seed(0); val = random.sample(val, min(300, len(val)))
from PIL import Image
pos_area = []
for r in val:
    _, a = seg.segment(Image.open(DS/r["image"]).convert("RGB"))
    pos_area.append(a)
pos_area = np.array(pos_area)

# NEGATIVES: extract 3 frames/clip, measure mask area
meta = json.load(open(Path.home()/"neg_meta.json"))
neg = {"reflection": [], "absent": []}
for m in meta:
    clip = Path.home()/os.path.relpath(m["path"]) if not os.path.isabs(m["path"]) else Path(m["path"])
    # clip path is repo-relative; on A100 the mp4s are under ~/seg_neg mirroring the tree
    cand = list((Path.home()/"seg_neg").rglob(Path(m["path"]).name))
    if not cand: continue
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg","-v","error","-i",str(cand[0]),"-vf","fps=0.2","-frames:v","3",
                        f"{td}/f%02d.jpg"], check=False)
        for f in sorted(glob.glob(f"{td}/*.jpg")):
            _, a = seg.segment(Image.open(f).convert("RGB"))
            neg[m["kind"]].append(a)
for k in neg: neg[k] = np.array(neg[k])
neg_all = np.concatenate([neg["reflection"], neg["absent"]])

def auc(pos, negs):  # rank-based AUC of area as present-detector
    lab = np.r_[np.ones(len(pos)), np.zeros(len(negs))]; sc = np.r_[pos, negs]
    order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc)+1)
    npos = lab.sum(); return (ranks[lab == 1].sum() - npos*(npos+1)/2) / (npos*(len(sc)-npos))

print(f"\nPRESENT (n={len(pos_area)}):   area median={np.median(pos_area):.3f} mean={pos_area.mean():.3f}")
print(f"REFLECTION (n={len(neg['reflection'])}): area median={np.median(neg['reflection']):.3f} mean={neg['reflection'].mean():.3f}")
print(f"ABSENT (n={len(neg['absent'])}):     area median={np.median(neg['absent']):.3f} mean={neg['absent'].mean():.3f}")
print(f"\nAUC (area separates present vs all-neg): {auc(pos_area, neg_all):.3f}")
print(f"AUC vs reflection only: {auc(pos_area, neg['reflection']):.3f}  | vs absent only: {auc(pos_area, neg['absent']):.3f}")
print("\nthreshold sweep (area >= t => 'octopus present'):")
print(f"{'t':>6} {'present_recall':>15} {'refl_FP':>9} {'absent_FP':>10}")
for t in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
    print(f"{t:6.3f} {(pos_area>=t).mean():15.2f} {(neg['reflection']>=t).mean():9.2f} {(neg['absent']>=t).mean():10.2f}")
