import os
import large_image
from pathlib import Path
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from torch import autocast
import torch.nn.functional as F
import time
from datetime import timedelta
import psutil
import platform
from loadmodel import load_model 
import argparse

parser = argparse.ArgumentParser(description="WSI inference with Segformer")
parser.add_argument("--model-type", type=str, default="lora",
                    help="Model type: pretrained | finetuned_full | finetuned_lastlayer | lora | qlora")
parser.add_argument("--model-dir", type=str, default=None,
                    help="Path to model directory (contains config.json) or adapter dir for LoRA/QLoRA")
parser.add_argument("--device", type=str, default=None,
                    help="Device to use: cuda or cpu (default: cuda if available)")
parser.add_argument("--wsi-path", type=str, default="/home/ajinkya/BS_two/data/iou_wsi/NA4972-02_AB17-24.svs",
                    help="Path to WSI file")
parser.add_argument("--out", type=str, default="output_inference.png")
args = parser.parse_args()

if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_type == "qlora" and device.type == "cpu":
    raise SystemExit("QLoRA requires CUDA device. Set --device cuda or choose another model type.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device.type == "cuda" else ""
print(f"Using device: {device}, model_type: {args.model_type}, model_dir: {args.model_dir}")

COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 255, 0],
    2: [0, 255, 255]
}


# determine model_dir default if not provided
model_dir = args.model_dir 
print(f"Loading model from: {model_dir}")

model = load_model(args.model_type, device, model_dir=model_dir)
wsi_path = Path(args.wsi_path)
ts = large_image.getTileSource(wsi_path, format='openslide')
metadata = ts.getMetadata()
tile_size = 512
xys = [
    (x, y)
    for x in range(0, metadata['sizeX'], tile_size)
    for y in range(0, metadata['sizeY'], tile_size)
]
transform = transforms.Compose([
    transforms.Resize((tile_size, tile_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
wsi_segmentation_map = np.zeros((metadata['sizeY'], metadata['sizeX']), dtype=np.uint8)
batch_size = 8
idx = list(range(0, len(xys), batch_size))


start_time = time.time()
for i in idx:
    batch_xys = xys[i:i + batch_size]
    batch_tiles = []
    for x, y in batch_xys:
        print(f"Processing tile at ({x}, {y})")
        region_width = min(tile_size, metadata['sizeX'] - x)
        region_height = min(tile_size, metadata['sizeY'] - y)
        tile = ts.getRegion(
            region=dict(left=x, top=y, width=region_width, height=region_height),
            format=large_image.tilesource.TILE_FORMAT_NUMPY
        )[0]
        tile_image = Image.fromarray(tile).resize((tile_size, tile_size))
        if tile_image.mode == 'RGBA':
            tile_image = tile_image.convert('RGB')
        tile_tensor = transform(tile_image).unsqueeze(0)
        batch_tiles.append(tile_tensor)
    batch_tensor = torch.cat(batch_tiles, dim=0).to(device)
    with torch.inference_mode(), autocast(device_type="cuda"):
        output = model(batch_tensor).logits
        output = F.interpolate(output, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        predicted_labels = output.argmax(dim=1).cpu().numpy()
    for idx_, (x, y) in enumerate(batch_xys):
        region_width = min(tile_size, metadata['sizeX'] - x)
        region_height = min(tile_size, metadata['sizeY'] - y)
        wsi_segmentation_map[y:y + region_height, x:x + region_width] = predicted_labels[idx_][:region_height, :region_width]

end_time = time.time()

# Log memory and time
process = psutil.Process(os.getpid())
cpu_mem = process.memory_info().rss / (1024 ** 2)
print(f"CPU memory used: {cpu_mem:.2f} MB")
print(f"Total time taken for WSI inference: {str(timedelta(seconds=end_time - start_time))}")


wsi_rgb_mask = np.zeros(
    (wsi_segmentation_map.shape[0], wsi_segmentation_map.shape[1], 3), dtype=np.uint8
)
for class_id, color in COLOR_MAP.items():
    wsi_rgb_mask[wsi_segmentation_map == class_id] = color
wsi_rgb_mask_image = Image.fromarray(wsi_rgb_mask)
wsi_rgb_mask_image.save("output_inference.png")
