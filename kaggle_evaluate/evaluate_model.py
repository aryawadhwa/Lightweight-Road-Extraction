import os
import subprocess
import shutil
import glob
import random
import sys
import cv2
import numpy as np

# 1. Setup Backend
backend_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'requirements.txt' in files and 'src' in dirs and 'scripts' in dirs:
        backend_dir = root
        break

working_backend = "/kaggle/working/backend"
if os.path.exists(working_backend):
    shutil.rmtree(working_backend)
shutil.copytree(backend_dir, working_backend)

print("Skipping pip install - Kaggle has pre-installed packages.")
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, working_backend)
from src.models.mobilevit_v2 import MobileViT_v2

# 2. Setup Dataset Split
deepglobe_train = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train' in dirs and deepglobe_train is None:
        try:
            if any('sat.jpg' in f for f in os.listdir(os.path.join(root, 'train'))):
                deepglobe_train = os.path.join(root, 'train')
        except: pass

if not deepglobe_train:
    deepglobe_train = "/kaggle/input/deepglobe-road-extraction-dataset/train"

working_dataset = "/kaggle/working/dataset"
valid_split = os.path.join(working_dataset, "valid")
os.makedirs(valid_split, exist_ok=True)

all_sats = glob.glob(os.path.join(deepglobe_train, "*_sat.jpg"))
all_ids = [os.path.basename(f).replace("_sat.jpg", "") for f in all_sats]
valid_ids = [i for i in all_ids if os.path.exists(os.path.join(deepglobe_train, f"{i}_mask.png"))]

random.seed(42)
random.shuffle(valid_ids)
split_idx = int(len(valid_ids) * 0.9)
val_ids = valid_ids[split_idx:]

for i in val_ids:
    os.symlink(os.path.join(deepglobe_train, f"{i}_sat.jpg"), os.path.join(valid_split, f"{i}_sat.jpg"))
    os.symlink(os.path.join(deepglobe_train, f"{i}_mask.png"), os.path.join(valid_split, f"{i}_mask.png"))

# 3. Load Model
model_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'best_model_v2.pth' in files:
        model_path = os.path.join(root, 'best_model_v2.pth')
        break

if not model_path:
    raise FileNotFoundError("Could not find best_model_v2.pth from the training kernel.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileViT_v2(num_classes=1, width_mult=1.0)

print(f"Loading weights from {model_path} onto {device}...")
state_dict = torch.load(model_path, map_location=device)
if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 4. Generate Predictions for the entire validation set
preds_dir = "/kaggle/working/outputs/preds"
os.makedirs(preds_dir, exist_ok=True)

transform = A.Compose([
    A.Resize(height=1024, width=1024),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
    ToTensorV2()
])

print(f"Generating predictions for {len(val_ids)} images...")
from tqdm import tqdm
with torch.no_grad():
    for img_id in tqdm(val_ids):
        img_path = os.path.join(deepglobe_train, f"{img_id}_sat.jpg")
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        
        # We need to process in full 1024x1024 to match how validation was run
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        prob_img = (probs * 255).astype(np.uint8)
        pred_mask = (probs > 0.5).astype(np.uint8) * 255
        
        cv2.imwrite(os.path.join(preds_dir, f"{img_id}_mask.png"), pred_mask)

# 5. Run evaluate.py
metrics_dir = "/kaggle/working/outputs/metrics"
os.makedirs(metrics_dir, exist_ok=True)
print("Running APLS and Topology evaluate.py...")
cmd = f"python {working_backend}/scripts/evaluate.py --gt_dir {valid_split} --pred_dir {preds_dir} --output_dir {metrics_dir} --no_wandb"
subprocess.run(cmd, shell=True, check=True)

# 6. Generate QGIS Heatmap for 3 sample images
vis_dir = "/kaggle/working/outputs/vis"
os.makedirs(vis_dir, exist_ok=True)

sample_ids = val_ids[:3]
for idx, img_id in enumerate(sample_ids):
    img_path = os.path.join(deepglobe_train, f"{img_id}_sat.jpg")
    gt_path = os.path.join(deepglobe_train, f"{img_id}_mask.png")
    
    img = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    prob_img = (probs * 255).astype(np.uint8)
    
    # Heatmap
    heatmap = cv2.applyColorMap(prob_img, cv2.COLORMAP_INFERNO)
    
    cv2.imwrite(os.path.join(vis_dir, f"sample_{idx}_{img_id}_original.jpg"), img)
    cv2.imwrite(os.path.join(vis_dir, f"sample_{idx}_{img_id}_gt.png"), gt_mask)
    cv2.imwrite(os.path.join(vis_dir, f"sample_{idx}_{img_id}_pred.png"), (probs > 0.5).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(vis_dir, f"sample_{idx}_{img_id}_heatmap.png"), heatmap)

print("Evaluation and Visualization Complete!")
