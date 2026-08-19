import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

NUM_CLASSES = 6

class WSIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        # Filter to valid image+label pairs: skip corrupt/empty/missing files.
        # os.listdir includes everything; some tiles may be <100 bytes or
        # missing their mask counterpart.
        all_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
        self.image_files = []
        skipped = 0
        for fname in all_files:
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(label_dir, fname.replace(".png", "_mask.png"))
            if not os.path.exists(mask_path):
                skipped += 1
                continue
            try:
                if os.path.getsize(img_path) < 100 or os.path.getsize(mask_path) < 100:
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue
            self.image_files.append(fname)
        if skipped:
            print(f"[data] filtered out {skipped} corrupt/missing tiles, "
                  f"{len(self.image_files)} valid")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".png", "_mask.png"))
        label = Image.open(label_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(np.array(label), dtype=torch.long)
        return image, label


def _scan_mask(path):
    """Worker: read one label mask, return per-class pixel counts."""
    try:
        a = np.array(Image.open(path).convert("L"), dtype=np.int64)
        vals, counts = np.unique(a, return_counts=True)
        cc = np.zeros(NUM_CLASSES, dtype=np.int64)
        for v, c in zip(vals.tolist(), counts.tolist()):
            if 0 <= v < NUM_CLASSES:
                cc[v] += c
        return cc
    except Exception:
        return np.zeros(NUM_CLASSES, dtype=np.int64)


def compute_class_weights(label_dir, num_classes=NUM_CLASSES, workers=8):
    """Scan all label masks in label_dir and return inverse-frequency class
    weights as a torch.FloatTensor.

    Background (class 0) is rare-by-pixel but a junk/dump class we want to
    suppress, so its weight is capped at 0.5 (not inflated by inverse-freq).
    Weights are normalized to mean 1.0.
    """
    label_files = sorted([
        os.path.join(label_dir, f)
        for f in os.listdir(label_dir)
        if f.endswith("_mask.png")
    ])
    if not label_files:
        raise ValueError(f"No label masks (*_mask.png) found in {label_dir}")

    class_counts = np.zeros(num_classes, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        for cc in ex.map(_scan_mask, label_files, chunksize=200):
            class_counts += cc

    total_px = int(class_counts.sum())
    class_names = ["BG", "WM", "GM", "Superficial", "Leptomeninges", "Exclude"]
    print(f"[data] class distribution ({total_px:,} px across {len(label_files)} masks):")
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"  {c} {name:14s}: {int(class_counts[c]):>13,} "
              f"({100*class_counts[c]/max(total_px,1):5.2f}%)")

    counts_safe = np.maximum(class_counts.astype(np.float64), 1.0)
    weights = total_px / (num_classes * counts_safe)
    weights[0] = min(weights[0], 0.5)
    weights = weights / weights.mean()
    print(f"[data] class weights: "
          + "  ".join(f"{class_names[c]}={weights[c]:.3f}"
                      for c in range(num_classes) if c < len(class_names)))
    return torch.tensor(weights, dtype=torch.float32)