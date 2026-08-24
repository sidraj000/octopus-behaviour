"""segment_octopus.py — inference for the tiny deployed octopus mask model.

Loads a checkpoint from train_segmenter.py and runs the compact U-Net on a frame.
This is the module the extraction gate (Phase 3) calls:

    from segment_octopus import OctoSegmenter
    seg = OctoSegmenter("weights/octo_seg_v1_ch16.pt")   # device auto
    mask, area = seg.segment(pil_or_bgr_frame)           # mask: HxW bool at native res, area: float

`area` (mask fraction of frame) is the presence signal; `mask` gates motion to
octopus-only pixels. Keeps the largest connected component so stray specks don't count.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train_segmenter import build_model, IMAGENET_MEAN, IMAGENET_STD


def _largest_blob(mask):
    try:
        import cv2
    except Exception:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


class OctoSegmenter:
    def __init__(self, ckpt, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available() else "cpu")
        c = torch.load(ckpt, map_location=self.device)
        self.in_size = c.get("in_size", 256)
        self.arch = c.get("arch", "unet")
        self.model = build_model(self.arch, c.get("base_ch", 16)).to(self.device).eval()
        self.model.load_state_dict(c["state_dict"])
        self.val = c.get("val")

    @staticmethod
    def _to_rgb(frame):
        if isinstance(frame, Image.Image):
            return np.asarray(frame.convert("RGB"))
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return np.stack([arr] * 3, -1)
        return arr[:, :, ::-1] if arr.shape[2] == 3 else arr[:, :, :3][:, :, ::-1]  # BGR->RGB

    @torch.no_grad()
    def segment(self, frame, thresh=0.5, largest_only=True):
        """Return (mask HxW bool at the frame's native size, area fraction)."""
        rgb = self._to_rgb(frame)
        H, W = rgb.shape[:2]
        x = np.asarray(Image.fromarray(rgb).resize((self.in_size, self.in_size), Image.BILINEAR),
                       np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        logits = self.model(t)
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        mask = (torch.sigmoid(up)[0, 0] > thresh).cpu().numpy()
        if largest_only and mask.any():
            mask = _largest_blob(mask)
        return mask, float(mask.mean())

    @torch.no_grad()
    def prob(self, frame):
        """Return the sigmoid probability map at the model's internal resolution (in_size x in_size).

        Exposed for TEMPORAL SMOOTHING: a caller can keep a running EMA of this map across frames
        (`ema = a*prob + (1-a)*ema`) and threshold the smoothed map — this removes per-frame jitter
        while the octopus's real motion still comes through. Cheap (low-res, one forward pass).
        """
        rgb = self._to_rgb(frame)
        x = np.asarray(Image.fromarray(rgb).resize((self.in_size, self.in_size), Image.BILINEAR),
                       np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        return torch.sigmoid(self.model(t))[0, 0].cpu().numpy()  # (in_size, in_size) float32

    @torch.no_grad()
    def segment_batch(self, frames, thresh=0.5, largest_only=True):
        """Batched version of segment() — identical per-frame result, one GPU forward for the whole
        batch. `frames` must all be the SAME size. Returns [(mask, area), ...]."""
        rgbs = [self._to_rgb(f) for f in frames]
        H, W = rgbs[0].shape[:2]
        xs = []
        for rgb in rgbs:
            x = np.asarray(Image.fromarray(rgb).resize((self.in_size, self.in_size), Image.BILINEAR),
                           np.float32) / 255.0
            xs.append(((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        t = torch.from_numpy(np.stack(xs)).to(self.device)
        up = F.interpolate(self.model(t), size=(H, W), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(up)[:, 0].cpu().numpy()   # (B, H, W)
        out = []
        for p in probs:
            m = p > thresh
            if largest_only and m.any():
                m = _largest_blob(m)
            out.append((m, float(m.mean())))
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the tiny segmenter on an image, save an overlay.")
    ap.add_argument("ckpt"); ap.add_argument("image"); ap.add_argument("--out", default="seg_overlay.png")
    a = ap.parse_args()
    seg = OctoSegmenter(a.ckpt)
    print(f"loaded {a.ckpt}  val={seg.val}")
    img = Image.open(a.image).convert("RGB")
    mask, area = seg.segment(img)
    print(f"area fraction = {area:.4f}")
    ov = np.asarray(img).copy()
    ov[mask] = (0.5 * ov[mask] + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(ov).save(a.out)
    print(f"overlay -> {a.out}")
