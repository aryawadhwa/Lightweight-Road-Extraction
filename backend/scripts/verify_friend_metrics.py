"""
verify_friend_metrics.py
Standalone fast validation verification script.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from backend.src.models.mobilevit_v2 import MobileViT_v2
from backend.src.utils.metrics_iou import compute_iou, compute_f1

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

def compute_relaxed_f1(pred_bin, gt_bin, slack=3):
    """
    Relaxed F1 score within a spatial slack tolerance buffer.
    """
    p = (pred_bin > 0).astype(np.uint8)
    g = (gt_bin > 0).astype(np.uint8)
    
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * slack + 1, 2 * slack + 1))
    g_dilated = cv2.dilate(g, kernel)
    p_dilated = cv2.dilate(p, kernel)

    # Relaxed precision: predicted pixels that fall within ground truth buffer
    prec_num = np.logical_and(p, g_dilated).sum()
    prec = prec_num / (p.sum() + 1e-7)

    # Relaxed recall: ground truth pixels that fall within predicted buffer
    rec_num = np.logical_and(g, p_dilated).sum()
    rec = rec_num / (g.sum() + 1e-7)

    f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
    return float(f1)

def compute_cldice_numpy(pred_bin, gt_bin):
    """
    Computes clDice metric on binary masks using skeletonization.
    """
    p = (pred_bin > 0).astype(bool)
    g = (gt_bin > 0).astype(bool)

    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    if skeletonize is not None:
        skel_p = skeletonize(p)
        skel_g = skeletonize(g)
    else:
        # Morphological thinning fallback
        skel_p = p
        skel_g = g

    tprec = (skel_p & g).sum() / (skel_p.sum() + 1e-7)
    tsens = (skel_g & p).sum() / (skel_g.sum() + 1e-7)

    cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-7)
    return float(cldice)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def evaluate_val_set(num_samples=20):
    val_dir = REPO_ROOT / "data" / "dataset" / "val_split"
    model_path = REPO_ROOT / "models" / "mobilevit_v2_best.pth"
    
    if not val_dir.exists():
        print(f"[ERROR] Val dir not found: {val_dir}")
        return
    if not model_path.exists():
        print(f"[ERROR] Model path not found: {model_path}")
        return

    device = get_device()
    print(f"[Device] Using {device} for evaluation...")
    
    model = MobileViT_v2(num_classes=1, width_mult=1.0)
    state = torch.load(model_path, map_location=device)
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    sat_files = sorted(list(val_dir.glob("*_sat.jpg")))[:num_samples]
    print(f"[Eval] Running verification on {len(sat_files)} validation image pairs at native resolution...")

    ious = []
    f1s = []
    relaxed_f1s = []
    cldices = []

    start_time = time.time()
    for idx, sat_path in enumerate(sat_files):
        mask_path = sat_path.parent / (sat_path.stem.replace("_sat", "") + "_mask.png")
        if not mask_path.exists():
            continue

        img_bgr = cv2.imread(str(sat_path))
        if img_bgr is None:
            continue
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # Preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_rgb - mean) / std
        tensor_in = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

        # 1. 4-Flip Test-Time Augmentation (TTA)
        with torch.no_grad():
            p1 = torch.sigmoid(model(tensor_in))
            p2 = torch.sigmoid(model(torch.flip(tensor_in, [2]))).flip([2])
            p3 = torch.sigmoid(model(torch.flip(tensor_in, [3]))).flip([3])
            p4 = torch.sigmoid(model(torch.flip(tensor_in, [2, 3]))).flip([2, 3])
            probs = ((p1 + p2 + p3 + p4) / 4.0).squeeze().cpu().numpy()

        # 2. Hysteresis thresholding (0.35 high, 0.12 low)
        high_mask = (probs >= 0.35).astype(np.uint8)
        low_mask = (probs >= 0.12).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(low_mask, connectivity=8)
        pred_binary = np.zeros_like(low_mask)
        for i in range(1, num_labels):
            if np.any(high_mask[labels == i]):
                pred_binary[labels == i] = 255

        # 3. 5x5 morphological gap closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel_close)

        gt_binary = (gt_mask > 127).astype(np.uint8) * 255

        iou_val = compute_iou(pred_binary, gt_binary)
        f1_val = compute_f1(pred_binary, gt_binary)
        rel_f1 = compute_relaxed_f1(pred_binary, gt_binary, slack=3)
        cldice_val = compute_cldice_numpy(pred_binary, gt_binary)

        ious.append(iou_val)
        f1s.append(f1_val)
        relaxed_f1s.append(rel_f1)
        cldices.append(cldice_val)

        if (idx + 1) % 5 == 0 or idx == len(sat_files) - 1:
            print(f"  [{idx+1:02d}/{len(sat_files)}] Current Mean -> Strict IoU: {np.mean(ious):.4f} | Strict F1: {np.mean(f1s):.4f} | Rel F1: {np.mean(relaxed_f1s):.4f} | clDice: {np.mean(cldices):.4f}")

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("LIVE RE-VERIFIED EVALUATION BENCHMARK")
    print("="*60)
    print(f"Evaluated Samples: {len(ious)} full 1024x1024 tiles")
    print(f"Total Time Taken:  {elapsed:.2f} seconds ({elapsed/len(ious):.3f} s/image)")
    print(f"Measured Strict IoU:       {np.mean(ious):.4f}  vs  Friend's: 0.5355")
    print(f"Measured Strict F1:        {np.mean(f1s):.4f}  vs  Friend's: 0.6975")
    print(f"Measured Relaxed F1 (@3px): {np.mean(relaxed_f1s):.4f}  vs  Friend's: 0.8084")
    print(f"Measured clDice:           {np.mean(cldices):.4f}  vs  Friend's: 0.7782")
    print("="*60)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    evaluate_val_set(num_samples=n)
