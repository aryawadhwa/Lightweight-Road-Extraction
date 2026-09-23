import os

import cv2
import numpy as np
from skimage.morphology import skeletonize


def simulate_weak_labels_deepglobe(input_mask_dir, output_weak_dir):
    """
    Takes standard thick pixel masks and skeletonizes them into 1-pixel centerlines
    to simulate weak labels for training.
    """
    if not os.path.exists(output_weak_dir):
        os.makedirs(output_weak_dir)

    masks = [f for f in os.listdir(input_mask_dir) if f.endswith("_mask.png")]

    for mask_name in masks:
        input_path = os.path.join(input_mask_dir, mask_name)
        output_path = os.path.join(output_weak_dir, mask_name)

        # 1. Load the thick DeepGlobe mask in grayscale
        thick_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 2. Convert to strict binary (0 and 1) for the skeletonize algorithm
        binary_mask = (thick_mask > 127).astype(np.uint8)

        # 3. Crush the road down to its absolute 1-pixel centerline
        skeleton = skeletonize(binary_mask)

        # 4. Convert back to image format (0 and 255) and save
        weak_label = (skeleton * 255).astype(np.uint8)
        cv2.imwrite(output_path, weak_label)

    print(f"Successfully generated {len(masks)} weak labels in {output_weak_dir}")

