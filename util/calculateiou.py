'''script to calculate iou'''

import numpy as np
from PIL import Image

# Increase the decompression bomb limit
Image.MAX_IMAGE_PIXELS = None  # Removes the limit (use with caution)

def compute_iou(gt_mask, pred_mask):
    """Compute Intersection over Union (IoU)."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else np.nan

COLOR_MAP = {
    
    "WM": [255, 255, 0],  # Yellow
    "GM": [0, 255, 255], # Cyan 
    "BG": [0, 0, 0]      # Black
}

# Paths
GROUND_TRUTH_PATHS = {
    "BG": "/home/ajinkya/BS_two/data/gt/NA4972-02_AB17-24-Background.png",
    "WM": "/home/ajinkya/BS_two/data/gt/NA4972-02_AB17-24-White.png",
    "GM": "/home/ajinkya/BS_two/data/gt/NA4972-02_AB17-24-Gray.png"
    # "BG": "/cache/Luca/Datasets/WMGM_data/gt/box_control_groundtruth/groundtruth/NA4894-02_AB17-24-Background.png",
    # "WM": "/cache/Luca/Datasets/WMGM_data/gt/box_control_groundtruth/groundtruth/NA4894-02_AB17-24-White.png",
    # "GM": "/cache/Luca/Datasets/WMGM_data/gt/box_control_groundtruth/groundtruth/NA4894-02_AB17-24-Gray.png"
}

PREDICTION_PATH = "/home/ajinkya/BS_two/src/squeezemodel/output_inference_onnx_cpu.png"
print("Prediction path:", PREDICTION_PATH)

reference_gt_path = list(GROUND_TRUTH_PATHS.values())[0]  
reference_gt_img = Image.open(reference_gt_path).convert("L")
gt_size = reference_gt_img.size  
print(gt_size)
pred_img = Image.open(PREDICTION_PATH).convert("RGB")
pred_img_size=pred_img.size
print(pred_img_size)
if(gt_size != pred_img_size):
    pred_img = pred_img.resize(gt_size, Image.NEAREST)  
print(pred_img.size)


# pred_img.save("/cache/Ajinkya/BS_optimize/data/brainseg/images/resized_vanilla.png")  # Save resized prediction mask
pred_img = np.array(pred_img)

# Compute IoU for each class
for class_name, color in COLOR_MAP.items():
    gt_img = Image.open(GROUND_TRUTH_PATHS[class_name]).convert("L")
    gt_mask = np.array(gt_img) > 128  # Thresholding

    pred_class_mask = np.all(pred_img == color, axis=-1)  # Extract class mask
    iou_score = compute_iou(gt_mask, pred_class_mask)

    print(f"IoU for {class_name}: {iou_score:.4f}")
