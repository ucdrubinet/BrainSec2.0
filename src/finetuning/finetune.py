import os
import time
import torch
import argparse
from data import WSIDataset, transform
from model import load_model, prepare_lora_model, prepare_qlora_model
from train import train, train_peft
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="Finetune Segformer (lora | qlora| standard)")
parser.add_argument("--mode", type=str, default="lora", choices=["lora", "qlora"],
                    help="Finetuning mode")
parser.add_argument("--model-path", type=str,
                    default=None,
                    help="Base model directory (contains config.json for HF models)")
parser.add_argument("--config-path", type=str, default=None,
                    help="Config path (optional; inferred from model-path if omitted)")
parser.add_argument("--image-dir", type=str,
                    default="/data/wsidir",
                    help="Directory with training tiles")
parser.add_argument("--label-dir", type=str,
                    default="/data/labeledmaskdir",
                    help="Directory with tile labels")
parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
parser.add_argument("--save-dir", type=str, default="ft_outputs",
                    help="Directory to save finetuned models / adapters")

args = parser.parse_args()

# device selection
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device.type == "cuda" else ""

print(f"Using device: {device}, mode: {args.mode}")

os.makedirs(args.save_dir, exist_ok=True)

# dataset + dataloader
train_dataset = WSIDataset(
    image_dir=args.image_dir,
    label_dir=args.label_dir,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# model selection / preparation
mode = args.mode
model_path = args.model_path
config_path = args.config_path or (os.path.join(model_path, "config.json") if model_path else None)

if mode == "lora":
    model = prepare_lora_model(model_path, config_path, device)
    train_fn = train_peft
elif mode == "qlora":
    model = prepare_qlora_model(model_path, config_path, device)
    train_fn = train_peft
else:
    # standard full fine-tune
    model = load_model(model_path, config_path, device)
    train_fn = train

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

start_time = time.time()
train_fn(model, train_loader, device, epochs=20)
end_time = time.time()
print(f"Total finetuning time: {end_time - start_time:.2f} seconds")

if mode == "lora":
    model.save_pretrained("finetuned_lora")
    print("Saved LoRA adapter weights.")
elif mode == "qlora":
    model.save_pretrained("finetuned_qlora")
    print("Saved QLoRA adapter weights.")
else:
    torch.save(model.state_dict(), "finetuned_full.pth")
    model.config.save_pretrained("finetuned_full")
    print("Saved model and config for inference!")