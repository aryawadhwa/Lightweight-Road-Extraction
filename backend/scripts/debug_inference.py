import torch
import cv2
import numpy as np
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
from src.models.mobilevit_v2 import MobileViT_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cpu")
model = MobileViT_v2(num_classes=1, width_mult=1.0)
state_dict = torch.load("models/best_model.pth", map_location=device, weights_only=True)
if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

img = cv2.imread("data/dataset/test/100393_sat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
    ToTensorV2()
])

img_tensor = transform(image=img_rgb)["image"].unsqueeze(0)

with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    
preds = (probs > 0.5).float()
pred_mask = preds.squeeze().cpu().numpy()

print(f"Probs > 0.5 count: {(probs > 0.5).sum().item()}")
print(f"Max prob: {probs.max().item():.4f}, Min prob: {probs.min().item():.4f}")

# Morphological
mask_uint8 = (pred_mask * 255).astype(np.uint8)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

print(f"Original thresholded mask max: {mask_uint8.max()}, unique: {np.unique(mask_uint8)}")
print(f"After CLOSE max: {mask_closed.max()}, unique: {np.unique(mask_closed)}")
print(f"After OPEN max: {mask_cleaned.max()}, unique: {np.unique(mask_cleaned)}")
