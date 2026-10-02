import os
import subprocess
import shutil

print("Files in /kaggle/input:")
for root, dirs, files in os.walk('/kaggle/input'):
    for d in dirs:
        print(f"DIR: {os.path.join(root, d)}")
    for f in files:
        if 'backend' in root or 'deepglobe' in root:
            # Skip printing every single image to keep logs clean
            continue
        print(f"FILE: {os.path.join(root, f)}")

print("Finding backend codebase...")
backend_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train.py' in files and 'dataset.py' in files and 'evaluate.py' in files:
        # We found the scripts directory or src directory? 
        pass
    if 'requirements.txt' in files and 'src' in dirs and 'scripts' in dirs:
        backend_dir = root
        break

if not backend_dir:
    raise FileNotFoundError("Could not find the backend codebase directory.")

print(f"Found backend at: {backend_dir}")

print("Copying backend to /kaggle/working/ to allow writes (logs/checkpoints)...")
working_backend = "/kaggle/working/backend"
if os.path.exists(working_backend):
    shutil.rmtree(working_backend)
shutil.copytree(backend_dir, working_backend)

print("Installing dependencies...")
subprocess.run(f"pip install -r {working_backend}/requirements.txt", shell=True, check=True)

print("Finding DeepGlobe dataset...")
deepglobe_train = None
deepglobe_valid = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train' in dirs and deepglobe_train is None:
        try:
            if any('sat.jpg' in f for f in os.listdir(os.path.join(root, 'train'))):
                deepglobe_train = os.path.join(root, 'train')
        except: pass
    if 'valid' in dirs and deepglobe_valid is None:
        try:
            if any('sat.jpg' in f for f in os.listdir(os.path.join(root, 'valid'))):
                deepglobe_valid = os.path.join(root, 'valid')
        except: pass

if not deepglobe_train:
    print("WARNING: Could not auto-detect DeepGlobe train directory. Falling back to default.")
    deepglobe_train = "/kaggle/input/deepglobe-road-extraction-dataset/train"
if not deepglobe_valid:
    print("WARNING: Could not auto-detect DeepGlobe valid directory. Falling back to default.")
    deepglobe_valid = "/kaggle/input/deepglobe-road-extraction-dataset/valid"

print("Starting training...")
cmd = f"""
python {working_backend}/scripts/train.py \
    --train_image_dir {deepglobe_train} \
    --train_mask_dir {deepglobe_train} \
    --val_image_dir {deepglobe_valid} \
    --val_mask_dir {deepglobe_valid} \
    --epochs 50 \
    --batch_size 16
"""

print(f"Running command: {cmd}")
subprocess.run(cmd, shell=True, check=True)
print("Training complete! You can download the model weights from the outputs tab.")
