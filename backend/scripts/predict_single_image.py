import argparse
import os
import cv2
import numpy as np
import torch
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Ensure backend directory is in PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from src.models.mobilevit_v2 import MobileViT_v2

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def postprocess_mask(mask_2d, min_area=50):
    """
    Applies aggressive morphology and annihilates micro-blobs (false positives).
    mask_2d is a boolean or uint8 numpy array (H, W).
    """
    mask_uint8 = (mask_2d * 255).astype(np.uint8)
    
    # Advanced Canopy Resilience Post-Processing (as per README)
    # 1. Close small gaps (trees, shadows)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Remove isolated noise blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Micro-Blob Annihilation (Connected Component Filter)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_cleaned[labels == i] = 0
            
    return mask_cleaned

def predict(image_path, model_path, output_path):
    device = get_device()
    print(f"[*] Using device: {device}")
    
    # Load model architecture
    # Defaulting width_mult=1.0 to match train.py default
    model = MobileViT_v2(num_classes=1, width_mult=1.0)
    
    # Load weights
    print(f"[*] Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Extract model_state_dict if it's a checkpoint dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    # Handle DataParallel 'module.' prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("[*] Model loaded successfully.")
    
    # Read Image
    print(f"[*] Reading image {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    
    # Apply Albumentations Normalization matching training
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    transformed = transform(image=img_rgb)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    print("[*] Running inference...")
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        
    probs_np = probs.squeeze().cpu().numpy()
    print("[*] Applying Hysteresis Thresholding (False Positive Filter)...")
    from src.utils.graph_postprocess import hysteresis_threshold
    pred_mask = hysteresis_threshold(probs_np, high_thresh=args.high_thresh, low_thresh=args.low_thresh)
    
    print("[*] Applying morphological post-processing and Micro-Blob Annihilation...")
    cleaned_mask = postprocess_mask(pred_mask, min_area=args.min_area)
    
    # Resize back to original dimensions
    final_mask = cv2.resize(cleaned_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_mask)
    print(f"[+] Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image with MobileViT")
    parser.add_argument("image_path", type=str, help="Path to input satellite image")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output mask (.png)")
    parser.add_argument("--high_thresh", type=float, default=0.6,
                        help="Hysteresis high threshold. Starts a road (e.g. 0.6)")
    parser.add_argument("--low_thresh", type=float, default=0.2,
                        help="Hysteresis low threshold. Continues a road (e.g. 0.2)")
    parser.add_argument("--min_area", type=int, default=50,
                        help="Annihilate blobs smaller than this pixel area")
    
    args = parser.parse_args()
    predict(args.image_path, args.model, args.output)
