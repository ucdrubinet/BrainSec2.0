import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def train(model, train_loader, device, epochs=20, class_weights=None,
          save_dir=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.decode_head.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=(512, 512),
                mode="nearest").squeeze(1).long()
            outputs = model(images).logits
            outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")


def train_peft(model, train_loader, device, epochs=20, class_weights=None,
               save_dir=None):
    """LoRA/QLoRA training loop with optional class-weighted loss and
    best-by-loss per-epoch checkpointing.

    Args:
        model: PEFT-wrapped model (LoRA or QLoRA).
        train_loader: DataLoader yielding (image, label) batches.
        device: torch device.
        epochs: number of epochs.
        class_weights: optional tensor of per-class weights for CrossEntropyLoss.
        save_dir: directory to save the best adapter checkpoint. If None,
                  checkpointing is skipped.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    log_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "training_log.csv")
        with open(log_path, "w") as lf:
            lf.write("epoch,avg_loss,saved_best\n")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=(512, 512),
                mode="nearest").squeeze(1).long()
            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                outputs = model(images).logits
                outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
                loss = criterion(outputs, labels)
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1

        avg_loss = train_loss / max(n_batches, 1)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            if save_dir:
                model.save_pretrained(save_dir)
                print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}  "
                      f"*saved best* -> {save_dir}")
            else:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}  "
                      f"*new best* (no checkpointing)")
        else:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}  "
                  f"(best={best_loss:.4f})")
        if log_path:
            with open(log_path, "a") as lf:
                lf.write(f"{epoch + 1},{avg_loss:.6f},{int(is_best)}\n")