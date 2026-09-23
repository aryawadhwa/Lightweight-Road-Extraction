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
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

# Save soft probability map
prob_img = (probs * 255).astype(np.uint8)
# Resize back to original size
prob_img = cv2.resize(prob_img, (img.shape[1], img.shape[0]))
cv2.imwrite("backend/outputs/prob_map.png", prob_img)

# Save lower threshold mask
preds_low = (probs > 0.05).astype(np.uint8) * 255
preds_low = cv2.resize(preds_low, (img.shape[1], img.shape[0]))
cv2.imwrite("backend/outputs/pred_low_thresh.png", preds_low)

print("Saved prob_map.png and pred_low_thresh.png")
