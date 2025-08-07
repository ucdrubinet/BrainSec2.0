import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def train(model, train_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.decode_head.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")

def train_peft(model, train_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
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
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}")