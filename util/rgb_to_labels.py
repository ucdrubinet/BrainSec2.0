import os
from PIL import Image
import numpy as np

# Define the color map
COLOR_MAP = {
    "BG": [0, 0, 0],       # Black -> Class 0
    "WM": [255, 255, 0],  # Yellow -> Class 1
    "GM": [0, 255, 255]   # Cyan -> Class 2
}

# Reverse the color map for easy lookup
COLOR_TO_CLASS = {tuple(color): class_id for class_id, color in enumerate(COLOR_MAP.values())}

def mask_to_class_labels(mask_path, output_path):
    # Load the mask
    mask = Image.open(mask_path).convert("RGB")
    mask_array = np.array(mask)

    # Create an empty array for class labels
    class_labels = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)

    # Map colors to class labels
    for color, class_id in COLOR_TO_CLASS.items():
        class_labels[np.all(mask_array == np.array(color), axis=-1)] = class_id

    # Save the class label map
    class_label_image = Image.fromarray(class_labels)
    class_label_image.save(output_path)


mask_dir = "/cache/Ajinkya/BS_optimize/data/finetune/maskdir"  # Directory containing RGB mask tiles
output_dir = "/cache/Ajinkya/BS_optimize/data/finetune/labeledmaskdir"  # Directory to save class label tiles
os.makedirs(output_dir, exist_ok=True)

# Process all mask tiles
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith(".png"):  # Process only PNG files
        mask_path = os.path.join(mask_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)
        mask_to_class_labels(mask_path, output_path)
        print(f"Processed {mask_file}")



###TEST:
# Load the original mask and class labels
mask = Image.open("/cache/Ajinkya/BS_optimize/data/finetune/labeledmaskdir/patch_1000_mask.png")
mask_array = np.array(mask)
print(mask_array.shape)

