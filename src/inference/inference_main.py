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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") # cpu does not work for QLoRA

COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 255, 0],
    2: [0, 255, 255]
}

# Check loadmodel file!
model_type = "qlora"  # Change to 'finetuned_full', 'finetuned_lastlayer', 'lora', or 'qlora' as needed
model_path="/home/ajinkya/segmentation/BrainSec2.0/models/ft_models/finetuned_qlora" #not needed for PEFT
model = load_model(model_type, device)

wsi_path = Path("/home/ajinkya/BS_two/data/iou_wsi/NA4972-02_AB17-24.svs")
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
