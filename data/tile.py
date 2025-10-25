'''script to tile wsis and save with a ID'''

import openslide
from PIL import Image
import os
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from a WSI and corresponding mask.")
    parser.add_argument("--wsi_path", default="/cache/Ajinkya/BS_optimize/data/finetune/wsis/NA4894-02_AB17-24.svs")
    parser.add_argument("--mask_path", default="/cache/Ajinkya/BS_optimize/data/finetune/masks/NA4894-02_AB17-24.png")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--output_image_dir", default="/cache/Ajinkya/BS_optimize/data/finetune/wsidir")
    parser.add_argument("--output_mask_dir", default="/cache/Ajinkya/BS_optimize/data/finetune/maskdir")

    args = parser.parse_args()

    extract_patches(args.wsi_path, args.mask_path, args.patch_size, args.output_image_dir, args.output_mask_dir)           
                
            