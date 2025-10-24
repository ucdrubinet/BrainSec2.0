# BrainSec2.0 — WSI segmentation toolkit

Run inference with SegFormer models on whole-slide-images (WSI).

## Features
- Dataset utilities and WSI tiling pipeline
- Finetuning: full-model, last-layer, LoRA and QLoRA workflows
- Quantization utilities (PyTorch dynamic quantization)
- Inference scripts for:
  - pretrained models
  - finetuned models (full / last-layer)
  - LoRA / QLoRA adapters
  - quantized models (CPU)
- Utilities split into modular files for reuse



## Models (stored on Google Drive)

Model weight files are large and are hosted on Google Drive (not in this repo). Before running training or inference, download the required model folders into the local `models/` directory so the code can find them from here: https://drive.google.com/drive/folders/1NbLP4E-m5RhgTmHj4mcee1ZR6OcevJIm?usp=sharing

```bash
mkdir -p models
unzip ~/Downloads/model.zip -d models/
```


## Quickstart

1. Create and Setup environment:
```
conda env create -f environment.yml
```
(If you don't have requirements.txt, install common packages: torch torchvision transformers peft large_image pillow matplotlib numpy tqdm bitsandbytes)

2. Prepare data (tiles / labeled masks)
- Place tiles in data/wsidir and corresponding labeled masks in data/labeledmaskdir
- For WSI inference, put slides in data/iou_wsi

3. Finetune
- Edit src/finetuning/finetune.py to set:
  - model_path (base model)
  - mode = "standard" | "lora" | "qlora"
  - data paths and batch size
- Run:
```bash
python src/finetuning/finetune.py
```
LoRA/QLoRA use PEFT; training function `train_peft` is in src/finetuning/train.py

4. Inference
- Use src/inference/loadmodel.py to load any model type. Example:
```python
from src.inference.loadmodel import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("finetuned_lastlayer", device, model_dir="/home/ajinkya/segmentation/BrainSec2.0/models/ft_models/finetuned_lastlayer", state_dict_path="<pth_path>")
```
- Run the appropriate inference script in src/inference. These scripts call shared utilities for tiling, batching, postprocess and saving masks.

5. Quantize
- Use src/quantize/quantize.py to quantize merged models and save quantized weights/config.
- For CPU inference, load config, instantiate model, call `torch.quantization.quantize_dynamic(...)` and load the quantized state_dict (see src/quantize/inference_quantized.py).

6. ONNX Conversion
- use src/onnxformatting/to_onnx.py to convert models to the onnx format
- use src/onnxformatting/inference_onnx.py to run inference (CPU or GPU) using the onnx model 

## Notes / best practices
- Do NOT commit large binary model files to git. Keep large weights in `models/` and add them to `.gitignore`.
- LoRA adapters are small — you can store adapters separately and apply to a base model at inference.
- QLoRA requires bitsandbytes and device-aware loading; QLoRA adapters are applied on top of a quantized base model using PEFT/PeftModel.


## Troubleshooting
- If running out of GPU memory, try:
  - smaller batch size
- If inference is slow on WSI tiling, increase batch size (GPU memory permitting) or downsample tiles.

