"""Build dataset v3 = colour positives (v1) + NEGATIVE frames (empty masks) from reflection/absent
clips, so the segmenter learns 'no octopus -> no mask' (fixes the presence-gate failure)."""
import glob, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

H = Path.home()
V1 = H / "dataset_seg/v1"
V3 = H / "dataset_seg/v3"
(V3 / "images").mkdir(parents=True, exist_ok=True)
(V3 / "masks").mkdir(parents=True, exist_ok=True)
N_PER = 4

# 1) hardlink positives + copy manifest
for img in glob.glob(str(V1 / "images/*.jpg")):
    d = V3 / "images" / os.path.basename(img)
    if not d.exists(): os.link(img, d)
for m in glob.glob(str(V1 / "masks/*.png")):
    d = V3 / "masks" / os.path.basename(m)
    if not d.exists(): os.link(m, d)
import shutil; shutil.copy(V1 / "manifest.jsonl", V3 / "manifest.jsonl")
npos = sum(1 for _ in open(V3 / "manifest.jsonl"))

# 2) extract negative frames, write empty masks, append manifest rows
clips = sorted(glob.glob(str(H / "seg_negtrain/**/*.mp4"), recursive=True))
def cam(p):
    for c in ("Right_Front","Right_Back","Right_Right","Right_Left","Right_Top"):
        if c in Path(p).name: return c
    return "?"
nneg = 0
with open(V3 / "manifest.jsonl", "a") as mf:
    for i, clip in enumerate(clips):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ffmpeg","-v","error","-i",clip,"-vf","fps=1","-frames:v","30",
                            f"{td}/f%03d.jpg"], check=False)
            fr = sorted(glob.glob(f"{td}/*.jpg"))
            if not fr: continue
            pick = [fr[k] for k in np.linspace(0, len(fr)-1, min(N_PER, len(fr))).astype(int)]
            for j, f in enumerate(sorted(set(pick))):
                im = Image.open(f).convert("RGB")
                stem = f"neg_{cam(clip)}_{i:04d}_{j}"
                im.save(V3 / "images" / f"{stem}.jpg", quality=90)
                Image.fromarray(np.zeros((im.size[1], im.size[0]), np.uint8)).save(V3 / "masks" / f"{stem}.png")
                mf.write(json.dumps({"clip": clip, "camera": cam(clip),
                                     "image": f"images/{stem}.jpg", "mask": f"masks/{stem}.png",
                                     "area": 0.0, "negative": True}) + "\n")
                nneg += 1
tot = sum(1 for _ in open(V3 / "manifest.jsonl"))
print(f"v3: {tot} pairs = {npos} positives + {nneg} negatives ({100*nneg/tot:.0f}% neg)")
