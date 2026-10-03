import os
import subprocess
import shutil
import glob
import random
import torch
import cv2
import numpy as np

print("Finding backend codebase...")
backend_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'requirements.txt' in files and 'src' in dirs and 'scripts' in dirs:
        backend_dir = root
        break

working_backend = "/kaggle/working/backend"
if os.path.exists(working_backend):
    shutil.rmtree(working_backend)
shutil.copytree(backend_dir, working_backend)

print("Finding DeepGlobe dataset...")
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

print(f"Validation set size: {len(val_ids)}")
for i in val_ids:
    os.symlink(os.path.join(deepglobe_train, f"{i}_sat.jpg"), os.path.join(valid_split, f"{i}_sat.jpg"))
    os.symlink(os.path.join(deepglobe_train, f"{i}_mask.png"), os.path.join(valid_split, f"{i}_mask.png"))

print("Running inference and generating masks...")
import sys
sys.path.insert(0, working_backend)
from src.models.mobilevit_v2 import MobileViT_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileViT_v2(num_classes=1, width_mult=1.0)

# We need to find the trained weights!
# Where are the weights? They are in another dataset we need to add!
# Wait! This script runs IN a Kaggle kernel. We cannot add the output of a previous Kaggle kernel to THIS run dynamically without uploading it as a dataset!
