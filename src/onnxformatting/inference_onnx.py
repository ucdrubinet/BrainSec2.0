''' ONNX inference script for WSI segmentation using Segformer '''
import os
import large_image
from pathlib import Path
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import time
from datetime import timedelta
import psutil
import onnxruntime as ort
import argparse


# Color map for RGB mask
COLOR_MAP = {
    0: [0, 0, 0],       # Black
    1: [255, 255, 0],   # Yellow
    2: [0, 255, 255]    # Cyan
}

# CLI argument parser
parser = argparse.ArgumentParser(description="ONNX WSI inference with Segformer")
parser.add_argument("--onnx-model", type=str, required=True,
                    help="Path to ONNX model file")
parser.add_argument("--wsi-path", type=str, required=True,
                    help="Path to WSI file")
parser.add_argument("--output", type=str, default="output_inference_onnx_cpu.png",
                    help="Output image path (default: output_inference_onnx_cpu.png)")
parser.add_argument("--executor-provider", type=str, default="CPUExecutionProvider",
                    choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                    help="ONNX execution provider (default: CPUExecutionProvider)")
parser.add_argument("--tile-size", type=int, default=512,
                    help="Tile size for processing (default: 512)")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for processing (default: 8)")
parser.add_argument("--scale-factor", type=float, default=1.0,
                    help="Scale factor for tile size (default: 1.0)")
args = parser.parse_args()

# Initialize ONNX session
ort_session = ort.InferenceSession(args.onnx_model, providers=[args.executor_provider])
print(f"Loaded ONNX model with provider: {args.executor_provider}")

device = torch.device("cpu")
def run_onnx_inference(batch_tensor):
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: batch_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outputs[0])  

# Load WSI using large_image
ts = large_image.getTileSource(args.wsi_path, format='openslide')
print(f"Loaded WSI: {args.wsi_path}")


# Image metadata
large_image_metadata = ts.getMetadata()
sf = args.scale_factor
tile_size = args.tile_size
fr_tile_size = int(tile_size / sf)  
xys = [
    (x, y)
    for x in range(0, large_image_metadata['sizeX'], fr_tile_size)
    for y in range(0, large_image_metadata['sizeY'], fr_tile_size)
]
print(f"{len(xys)} tiles to process")

transform = transforms.Compose([
    transforms.Resize((tile_size, tile_size)),  
    transforms.ToTensor(),                     
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

wsi_segmentation_map = np.zeros(
    (large_image_metadata['sizeY'], large_image_metadata['sizeX']), dtype=np.uint8
)
batch_size = args.batch_size
idx = list(range(0, len(xys), batch_size))
print(f"Processing {len(idx)} batches with batch size = {batch_size}")



# Log memory before
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 ** 2)  # MB

# Inference loop
print("Starting inference...")
start_time = time.time()
for i in idx:
    batch_xys = xys[i:i + batch_size]
    batch_tiles = []

    for x, y in batch_xys:
        region_width = min(fr_tile_size, large_image_metadata['sizeX'] - x)
        region_height = min(fr_tile_size, large_image_metadata['sizeY'] - y)
        tile = ts.getRegion(
            region=dict(left=x, top=y, width=region_width, height=region_height),
            format=large_image.tilesource.TILE_FORMAT_NUMPY
        )[0]
        tile_image = Image.fromarray(tile).resize((tile_size, tile_size))
        tile_image = tile_image.convert("RGB")
        tile_tensor = transform(tile_image).unsqueeze(0)
        batch_tiles.append(tile_tensor)
    batch_tensor = torch.cat(batch_tiles, dim=0)

    with torch.inference_mode():
        #print("Running ONNX inference...")
        output = run_onnx_inference(batch_tensor)  # returns torch.Tensor from ONNX output
        output = output.to(dtype=torch.float32)
        output = F.interpolate(output, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        predicted_labels = output.argmax(dim=1).cpu().numpy()  

    for idx, (x, y) in enumerate(batch_xys):
        print(f"Processing tile at ({x}, {y})")
        region_width = min(fr_tile_size, large_image_metadata['sizeX'] - x)
        region_height = min(fr_tile_size, large_image_metadata['sizeY'] - y)
        wsi_segmentation_map[y:y + region_height, x:x + region_width] = predicted_labels[idx][:region_height, :region_width]


end_time = time.time()
mem_after = process.memory_info().rss / (1024 ** 2)
mem_used = mem_after - mem_before
total_time = str(timedelta(seconds=end_time - start_time))
print(f"CPU memory used: {mem_used:.2f} MB")
print(f"Total inference time: {total_time}")

rgb_mask = np.zeros((wsi_segmentation_map.shape[0], wsi_segmentation_map.shape[1], 3), dtype=np.uint8)
for class_id, color in COLOR_MAP.items():
    rgb_mask[wsi_segmentation_map == class_id] = color
Image.fromarray(rgb_mask).save("output_inference_onnx_cpu.png")
