import cv2
import numpy as np

img = cv2.imread('backend/outputs/val_prob_map.png', cv2.IMREAD_GRAYSCALE)
if img is not None:
    # Min-max normalization
    min_val = img.min()
    max_val = img.max()
    normalized = ((img - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
    
    # Apply heatmap
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    cv2.imwrite('backend/outputs/val_heatmap.png', heatmap)
