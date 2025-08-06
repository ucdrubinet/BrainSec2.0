'''script to tile wsis and save with a ID'''
'''dont need x and y coordinates as we do inference with WSIs directly in finetuning experiments'''
'''use file inside sampath folder if need to stitch according to IDs'''

'''OPTIMAL: please add x and y coordinates to filenames'''

import openslide
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb check


def extract_patches(wsi_path, mask_path, patch_size, output_image_dir, output_mask_dir):
    wsi = openslide.OpenSlide(wsi_path)
    mask = Image.open(mask_path)
    wsi_width, wsi_height = wsi.dimensions

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    patch_id = 0
    for y in range(0, wsi_height, patch_size):
        for x in range(0, wsi_width, patch_size):
            patch = wsi.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert("RGB")  

            mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))

            patch.save(os.path.join(output_image_dir, f"patch_{patch_id}.png"))
            mask_patch.save(os.path.join(output_mask_dir, f"patch_{patch_id}_mask.png"))

            patch_id += 1


extract_patches(
    wsi_path="/cache/Ajinkya/BS_optimize/data/finetune/wsis/NA4894-02_AB17-24.svs",
    mask_path="/cache/Ajinkya/BS_optimize/data/finetune/masks/NA4894-02_AB17-24.png",
    patch_size=512,
    output_image_dir="/cache/Ajinkya/BS_optimize/data/finetune/wsidir",
    output_mask_dir="/cache/Ajinkya/BS_optimize/data/finetune/maskdir"
)