"""train_segmenter.py — Phase 2: train the tiny deployed octopus mask model.

Trains a compact single-class U-Net on the (image, mask) pairs produced by
`auto_segment.py` (the GroundingDINO+SAM2 teacher). This is the SMALL model that
actually ships in the extraction gate — the teacher never deploys.

Design (per SEGMENTATION_PLAN.md):
  * single class (octopus vs background), low input res (default 256).
  * smallest-first: a compact U-Net whose width is set by --base-ch. Sweep --base-ch
    (8/16/24/32) to trace the IoU-vs-size curve and pick the smallest model clearing
    the bar (val mask IoU >= 0.85 on colour cameras).
  * split BY SOURCE VIDEO (date/segment) so frames from one recording never straddle
    train/val — the honest generalization number.
  * loss = BCEWithLogits + soft Dice; metric = IoU@0.5 (+ Dice, mean area error).

Reads a dataset dir written by auto_segment.py:  <ds>/images/*.jpg, <ds>/masks/*.png,
<ds>/manifest.jsonl (one row per pair: image, mask, clip, camera, area, best_conf).

Saves weights/octo_seg_<ver>.pt = {state_dict, arch, base_ch, in_size, val: {...},
n_params, cameras}. Use src/segment_octopus.py to run inference from that checkpoint.

CLI:
  python3 train_segmenter.py --ds src/dataset_seg/v1 --base-ch 16 --epochs 40
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


# ── model: compact U-Net ──────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """4-level U-Net; width scales with base_ch (8->~0.13M, 16->~0.5M, 32->~2M params)."""
    def __init__(self, base_ch=16, in_ch=3):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.d1 = DoubleConv(in_ch, c[0])
        self.d2 = DoubleConv(c[0], c[1])
        self.d3 = DoubleConv(c[1], c[2])
        self.bott = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        xb = self.bott(self.pool(x3))
        y = self.u3(torch.cat([self.up3(xb), x3], 1))
        y = self.u2(torch.cat([self.up2(y), x2], 1))
        y = self.u1(torch.cat([self.up1(y), x1], 1))
        return self.head(y)  # logits [B,1,H,W]


class LRASPP(nn.Module):
    """torchvision LR-ASPP on a MobileNetV3-Large backbone (ImageNet-pretrained).

    A pretrained encoder dramatically improves localization on limited data vs. the
    from-scratch U-Net, at ~3.2M params (~3 MB INT8) — still within the deploy budget.
    """
    def __init__(self):
        super().__init__()
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        from torchvision.models import MobileNet_V3_Large_Weights
        net = lraspp_mobilenet_v3_large(num_classes=1,
                                        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]  # logits [B,1,H,W] at input resolution


def build_model(arch, base_ch):
    if arch == "unet":
        return TinyUNet(base_ch=base_ch)
    if arch == "lraspp":
        return LRASPP()
    raise ValueError(f"unknown arch {arch}")


def n_params_of(model):
    return sum(p.numel() for p in model.parameters())


# ── data ────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def source_video(clip_path):
    """Group key = the source recording: .../{date}/{segment}/Camera_s-e.mp4 -> 'date/segment'."""
    p = Path(clip_path)
    return f"{p.parent.parent.name}/{p.parent.name}"


class SegDS(Dataset):
    def __init__(self, rows, ds_root, size=256, train=False, aug="strong"):
        self.rows, self.root, self.size, self.train, self.aug = rows, Path(ds_root), size, train, aug

    def __len__(self):
        return len(self.rows)

    def _augment(self, img, m):
        """Strong spatial + photometric aug (image & mask kept in lock-step).

        Targets the v1 failure mode — the model finds an octopus-sized blob in the WRONG
        place — by teaching position/scale/rotation invariance. Affine keeps the (small)
        octopus mostly in-frame, unlike aggressive random-resized-crop which would blank it out.
        """
        import random as _r
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode as IM
        if _r.random() < 0.5:
            img = TF.hflip(img); m = TF.hflip(m)
        if self.aug == "strong":
            ang = _r.uniform(-15, 15)
            tr = [_r.uniform(-0.10, 0.10) * self.size, _r.uniform(-0.10, 0.10) * self.size]
            sc = _r.uniform(0.75, 1.25)
            img = TF.affine(img, angle=ang, translate=tr, scale=sc, shear=[0.0],
                            interpolation=IM.BILINEAR, fill=[128, 128, 128])
            m = TF.affine(m, angle=ang, translate=tr, scale=sc, shear=[0.0],
                          interpolation=IM.NEAREST, fill=[0])
        img = TF.adjust_brightness(img, _r.uniform(0.75, 1.25))
        img = TF.adjust_contrast(img, _r.uniform(0.75, 1.25))
        return img, m

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["image"]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        m = Image.open(self.root / r["mask"]).convert("L").resize((self.size, self.size), Image.NEAREST)
        if self.train and self.aug != "none":
            img, m = self._augment(img, m)
        img = np.asarray(img, np.float32) / 255.0
        m = (np.asarray(m, np.float32) > 127).astype(np.float32)
        if self.train and self.aug == "strong":              # mild sensor noise
            img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(img.transpose(2, 0, 1)),
                torch.from_numpy(m)[None])


# ── loss / metrics ────────────────────────────────────────────────────────────────
def dice_bce_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean()


def focal_tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=1.3333, eps=1.0):
    """Tversky (Dice generalized: penalize FN vs FP asymmetrically) + focal focus on hard examples.

    Our failure mode is UNDER-segmentation (missing thin tentacles + small/resting octopus = false
    negatives), so beta>alpha makes false negatives cost more -> the model reaches further into the
    faint/thin parts. gamma>1 focuses gradient on the low-overlap (hard) images. Add a light BCE term
    for stable pixel gradients. alpha=beta=0.5,gamma=1 recovers plain Dice."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum((1, 2, 3))
    fp = (p * (1 - target)).sum((1, 2, 3))
    fn = ((1 - p) * target).sum((1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    ft = ((1 - tversky) ** gamma).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return ft + 0.5 * bce


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious, dices, area_err = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p * y).sum((1, 2, 3))
        union = ((p + y) > 0).float().sum((1, 2, 3))
        iou = torch.where(union > 0, inter / union, torch.ones_like(union))  # both empty => perfect
        dice = torch.where((p.sum((1, 2, 3)) + y.sum((1, 2, 3))) > 0,
                           2 * inter / (p.sum((1, 2, 3)) + y.sum((1, 2, 3))), torch.ones_like(inter))
        ae = (p.mean((1, 2, 3)) - y.mean((1, 2, 3))).abs()
        ious += iou.tolist(); dices += dice.tolist(); area_err += ae.tolist()
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)),
            "area_err": float(np.mean(area_err)), "n": len(ious)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(HERE / "dataset_seg" / "v1"))
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--arch", default="unet", choices=["unet", "lraspp"])
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--in-size", type=int, default=256)
    ap.add_argument("--loss", default="dice_bce", choices=["dice_bce", "focal_tversky"],
                    help="focal_tversky penalizes false negatives more (beta>alpha) — targets the "
                         "under-segmentation failure mode (missing tentacles / small octopus)")
    ap.add_argument("--tversky-alpha", type=float, default=0.3, help="FP weight (lower = more permissive)")
    ap.add_argument("--tversky-beta", type=float, default=0.7, help="FN weight (higher = reach further)")
    ap.add_argument("--tversky-gamma", type=float, default=1.3333, help="focal power on hard examples")
    ap.add_argument("--holdout-videos", default="", help="file of source_video keys (date/segment) to "
                    "force TEST-only — excluded from training across ALL datasets (prevents leakage)")
    ap.add_argument("--aug", default="strong", choices=["strong", "basic", "none"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of SOURCE VIDEOS held out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources", default="", help="comma-list of manifest 'source' values to keep "
                    "(e.g. 'human' = positives only for a clean mask-IoU eval; '' = all)")
    ap.add_argument("--out", default=None, help="checkpoint path (default weights/octo_seg_<ver>_ch<base>.pt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    ds_root = Path(args.ds)
    rows = [json.loads(l) for l in open(ds_root / "manifest.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("image") and r.get("mask")]     # drop reject rows (no files)
    if args.sources:                                                  # e.g. "human" = positives only (mask IoU)
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r.get("source", "human") in keep]
    if not rows:
        raise SystemExit(f"no rows in {ds_root}/manifest.jsonl — run auto_segment.py first")

    # split by source video (no frame/video leakage)
    vids = sorted({source_video(r["clip"]) for r in rows})
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac))
    val_vids = set(vids[:n_val])
    if args.holdout_videos:                      # force specific source-videos to be TEST-only (never trained)
        hv = set(l.strip() for l in open(args.holdout_videos) if l.strip())
        val_vids |= hv
        print(f"forced holdout: {len(hv & set(vids))}/{len(hv)} holdout videos present -> excluded from train", flush=True)
    tr = [r for r in rows if source_video(r["clip"]) not in val_vids]
    va = [r for r in rows if source_video(r["clip"]) in val_vids]
    # NOTE: report the ACTUAL split. This used to print `len(vids)-n_val / n_val`, which ignored the
    # forced holdout and so under-reported val whenever --holdout-videos added videos to it (thin768
    # logged "train 147 / val 36" for a split that was really 142/41). The frame counts were always
    # right, so the numbers looked mutually inconsistent and cost an audit to re-derive.
    print(f"device={device}  pairs={len(rows)}  videos={len(vids)} "
          f"(train {len(vids)-len(val_vids)} / val {len(val_vids)})  ->  "
          f"train {len(tr)} / val {len(va)} frames", flush=True)

    tl = DataLoader(SegDS(tr, ds_root, args.in_size, train=True, aug=args.aug), batch_size=args.batch,
                    shuffle=True, num_workers=4, pin_memory=(device == "cuda"), drop_last=True)
    vl = DataLoader(SegDS(va, ds_root, args.in_size), batch_size=args.batch,
                    shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    model = build_model(args.arch, args.base_ch).to(device)
    n_params = n_params_of(model)
    tag = f"{args.arch}" + (f" base_ch={args.base_ch}" if args.arch == "unet" else "")
    print(f"{tag}: {n_params/1e6:.3f}M params (~{n_params*4/1e6:.1f} MB fp32)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_iou, best_state, best_metrics = -1.0, None, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = (focal_tversky_loss(model(x), y, args.tversky_alpha, args.tversky_beta,
                                           args.tversky_gamma) if args.loss == "focal_tversky"
                        else dice_bce_loss(model(x), y))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        m = evaluate(model, vl, device)
        print(f"ep {ep+1:3d}/{args.epochs}  loss {np.mean(losses):.4f}  "
              f"val IoU {m['iou']:.4f}  Dice {m['dice']:.4f}  areaErr {m['area_err']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if m["iou"] > best_iou:
            best_iou = m["iou"]; best_metrics = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    suffix = f"ch{args.base_ch}" if args.arch == "unet" else args.arch
    out = Path(args.out) if args.out else (REPO / "weights" / f"octo_seg_{args.ver}_{suffix}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": args.arch, "base_ch": args.base_ch,
                "in_size": args.in_size, "aug": args.aug, "epochs": args.epochs, "loss": args.loss,
                "val": best_metrics, "n_params": n_params, "ds": str(ds_root)}, out)
    bar = "PASS" if best_iou >= 0.85 else "below 0.85 bar"
    print(f"\nBEST val IoU {best_iou:.4f} ({bar})  |  {n_params/1e6:.3f}M params  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
