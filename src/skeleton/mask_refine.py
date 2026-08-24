"""mask_refine.py — OFFLINE mask refinement: the tiny student LOCATES, SAM2 SHARPENS.

Measured on the frozen bench50: thin768 base 4.60 arms/0.768 tip-match -> SAM2-refined
5.04/0.792 (better on BOTH). ~1-2 s/frame on MPS — offline/research-grade only, never the live gate.
(The zoom-2-pass alternative measured WORSE: crops are out-of-distribution for the student.)
"""
import cv2
import numpy as np

_SAM = None
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def sam2_refine(img_bgr, student_mask, largest_blob=None):
    """Refine a boolean/0-255 student mask with SAM2 (box + interior positive points). Returns bool."""
    global _SAM
    import torch
    if _SAM is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _SAM = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device=dev)
    m = np.asarray(student_mask) > 0
    if not m.any():
        return m
    ys, xs = np.where(m)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    pts, dd = [], dt.copy()
    for _ in range(5):
        y, x = np.unravel_index(np.argmax(dd), dd.shape)
        if dd[y, x] <= 1:
            break
        pts.append([x, y])
        cv2.circle(dd, (int(x), int(y)), max(8, int(dt.max() * 0.8)), 0, -1)
    _SAM.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    masks, _, _ = _SAM.predict(point_coords=np.array(pts, np.float32),
                               point_labels=np.ones(len(pts), np.int32),
                               box=box, multimask_output=False)
    out = masks[0].astype(bool)
    if largest_blob is not None and out.any():
        out = largest_blob(out)
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)
    return out
