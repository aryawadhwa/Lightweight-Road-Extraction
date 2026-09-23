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
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Test without normalization (what predict_single_image.py did)
img_resized = cv2.resize(img, (256, 256))
img_tensor_unnorm = torch.from_numpy(img_resized).float() / 255.0
img_tensor_unnorm = img_tensor_unnorm.permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    logits1 = model(img_tensor_unnorm)
    probs1 = torch.sigmoid(logits1)
    print(f"Unnormalized - Min prob: {probs1.min().item():.4f}, Max prob: {probs1.max().item():.4f}, Mean: {probs1.mean().item():.4f}")

# 2. Test with normalization (what it SHOULD be)
transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
    ToTensorV2(),
])
img_tensor_norm = transform(image=img)["image"].unsqueeze(0)

with torch.no_grad():
    logits2 = model(img_tensor_norm)
    probs2 = torch.sigmoid(logits2)
    print(f"Normalized - Min prob: {probs2.min().item():.4f}, Max prob: {probs2.max().item():.4f}, Mean: {probs2.mean().item():.4f}")

