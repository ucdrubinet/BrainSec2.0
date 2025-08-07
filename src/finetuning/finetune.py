import os
import time
import torch

from data import WSIDataset, transform
from model import load_model, prepare_lora_model, prepare_qlora_model
from train import train, train_peft
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = WSIDataset(
    image_dir="/home/ajinkya/BS_two/data/wsidir",
    label_dir="/home/ajinkya/BS_two/data/labeledmaskdir",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model_path = "/home/ajinkya/BS_two/models/models/segformer_JC"
config_path = "/home/ajinkya/BS_two/models/models/segformer_JC/config.json"

mode = "qlora"  # Change to "lora" or "qlora" as needed

if mode == "lora":
    model = prepare_lora_model(model_path, config_path, device)
    train_fn = train_peft
elif mode == "qlora":
    model = prepare_qlora_model(model_path, config_path, device)
    train_fn = train_peft
else:
    model = load_model(model_path, config_path, device)
    train_fn = train

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

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