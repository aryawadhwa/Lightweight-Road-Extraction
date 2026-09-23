import cv2
import numpy as np

mask = cv2.imread("backend/outputs/prediction_result.png", cv2.IMREAD_GRAYSCALE)
if mask is not None:
    print(f"Mask Min: {mask.min()}, Max: {mask.max()}, Unique values: {np.unique(mask)}")
else:
    print("Could not load mask")
