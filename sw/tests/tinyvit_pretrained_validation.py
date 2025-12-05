# %%
# basic pytorch test
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %%
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import re
import matplotlib.pyplot as plt

# Auto-detect data root (works when running from notebooks/ or project root)
_candidates = [
    Path("../../data/datasets/imagenet1k_val_subset/ILSVRC2012_img_val_subset")
]
ROOT = next((p for p in _candidates if p.exists()), None)
if ROOT is None:
    raise FileNotFoundError(f"Could not find dataset in any of: {_candidates} (CWD={Path.cwd()})")

CLASS_INDEX_PATH = ROOT.parent / "imagenet_class_index.json"
if not CLASS_INDEX_PATH.exists():
    raise FileNotFoundError(f"imagenet_class_index.json not found at {CLASS_INDEX_PATH}")

with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    class_index = json.load(f)  # keys like "000" -> [wnid, human_label]

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
def list_image_files(dirpath: Path):
    return sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXT])

# Standard ImageNet transforms for pretrained ResNet18
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class ImageNetSubsetDataset(Dataset):
    """ROOT/<class_dir>/*.jpg layout. Returns (img_tensor, label_int, class_str, human_label, path)."""
    def __init__(self, root: Path, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.samples = []
        digit_dir_re = re.compile(r"^\d{3}$")
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and digit_dir_re.match(d.name)])
        for d in class_dirs:
            for p in list_image_files(d):
                cls = d.name
                human = class_index.get(cls, ["", "(unknown)"])[1]
                self.samples.append((p, int(cls), cls, human))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, label_int, cls_str, human = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, label_int, cls_str, human, str(p)

def prepare_dataloader(batch_size=16, num_workers=0, shuffle=False, transforms=val_transforms):
    ds = ImageNetSubsetDataset(ROOT, transforms=transforms)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return dl

# Example: create dataloader for inference
val_loader = prepare_dataloader(batch_size=16, num_workers=0, shuffle=False)
print(f"Prepared DataLoader with {len(val_loader.dataset)} images.")

# Show first image from first 20 classes (000..019)
n_classes = 20
selected_imgs = []
titles = []

# val_loader.dataset.samples entries: (path, label_int, cls_str, human)
for i in range(n_classes):
    cls = f"{i:03d}"
    found = False
    for idx, sample in enumerate(val_loader.dataset.samples):
        if sample[2] == cls:
            img_tensor, label_int, cls_str, human, _ = val_loader.dataset[idx]  # __getitem__ applies transforms
            selected_imgs.append(img_tensor)
            titles.append(f"{cls} - {human}")
            found = True
            break
    if not found:
        selected_imgs.append(None)
        titles.append(f"{cls} (missing)")

# Stack only available tensors
available = [t for t in selected_imgs if t is not None]
if available:
    imgs_t = torch.stack(available)  # (N, C, H, W)
    # Denormalize
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    imgs_denorm = imgs_t * std + mean
    imgs_denorm = imgs_denorm.clamp(0.0, 1.0)
    imgs_np = imgs_denorm.permute(0, 2, 3, 1).cpu().numpy()
else:
    imgs_np = []

# Plot grid (placeholders for missing)
cols = 5
rows = (n_classes + cols - 1) // cols
plt.figure(figsize=(cols * 3, rows * 3))
img_i = 0
for i in range(n_classes):
    plt.subplot(rows, cols, i + 1)
    if selected_imgs[i] is None:
        plt.text(0.5, 0.5, titles[i], ha="center", va="center")
        plt.axis("off")
    else:
        plt.imshow(imgs_np[img_i])
        plt.title(titles[i], fontsize=8)
        plt.axis("off")
        img_i += 1
plt.tight_layout()
plt.show()

# %%
from pathlib import Path
import os
import torch
import timm

# -------------------------------
# 1. Models directory
# -------------------------------
models_dir = Path("../../data/models")
models_dir.mkdir(parents=True, exist_ok=True)

print("Using models directory:", models_dir.resolve())

# -------------------------------
# 2. Force HuggingFace + timm cache into models/
# -------------------------------
# Disable Xet Storage (important for TinyViT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# HF uses these paths; timm uses HF internally
os.environ["HF_HOME"] = str(models_dir.resolve())
os.environ["HF_HUB_CACHE"] = str(models_dir.resolve())

# (Optional but safe – prevent other caches)
os.environ["XDG_CACHE_HOME"] = str(models_dir.resolve())

# -------------------------------
# 3. Filenames for saving
# -------------------------------

# Original TinyViT-5M pretrained on ImageNet-1K
# state_fname = "tinyvit_5m_224_in1k_state_dict.pth"
# full_fname  = "tinyvit_5m_224_in1k_fullmodel.pth"
# TinyViT-5M with distillation pretrained on ImageNet-22K and fine-tuned on ImageNet-1K
state_fname = "tinyvit_5m_224_dist_in22k_ft_in1k_state_dict.pth"
full_fname  = "tinyvit_5m_224_dist_in22k_ft_in1k_fullmodel.pth"

state_path = models_dir / state_fname
full_path  = models_dir / full_fname

# -------------------------------
# 4. Load TinyViT
# -------------------------------

# model_name = "tiny_vit_5m_224.in1k"   # original name without distillation
model_name = "tiny_vit_5m_224.dist_in22k_ft_in1k"  # with distillation

if state_path.exists():
    print(f"Found saved TinyViT state_dict at {state_path.resolve()}")
    print("Building TinyViT architecture without pretrained weights...")
    model = timm.create_model(model_name, pretrained=False)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
else:
    print("Downloading / using TinyViT pretrained weights (via HF HUB)...")
    model = timm.create_model(model_name, pretrained=True)

model.eval()

# -------------------------------
# 5. Save state_dict and full model
# -------------------------------
try:
    torch.save(model.state_dict(), state_path)
    torch.save(model, full_path)

    print("\nSaved TinyViT files:")
    print(" - state_dict ->", state_path.resolve())
    print(" - full model ->", full_path.resolve())

except Exception as e:
    print("Error while saving TinyViT files:", e)

# -------------------------------
# 6. Print architecture summary
# -------------------------------
print("\nTinyViT-5M architecture:\n")
print(model)

# %%
# evaluate pretrained model on val_loader (Top1 / Top5)
import torch
try:
    from tqdm.auto import tqdm
    _tqdm_available = True
except Exception:
    tqdm = None
    _tqdm_available = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Reported accuracy — Top1: {79.1:.2f}%, Top5: {94.8:.2f}%")

if "val_loader" not in globals():
    print("val_loader not found in notebook.")
else:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    iterator = val_loader
    if _tqdm_available:
        iterator = tqdm(val_loader, desc="Evaluating", unit="batch", total=len(val_loader))
    else:
        print("tqdm not available — will print progress every 50 batches.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            images, labels, *rest = batch  # dataset returns (img, label, cls_str, human, path)
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)                      # (B, 1000)
            topk = logits.topk(5, dim=1).indices        # (B, 5)

            # Top-1
            pred1 = topk[:, 0]
            correct1 += (pred1 == labels).sum().item()

            # Top-5
            correct5 += (topk == labels.view(-1, 1)).any(dim=1).sum().item()

            total += labels.size(0)

            if not _tqdm_available and (batch_idx + 1) % 50 == 0:
                print(f"Processed {total} samples (batches: {batch_idx + 1}/{len(val_loader)})")

    if total == 0:
        print("No samples evaluated.")
    else:
        top1_acc = 100.0 * correct1 / total
        top5_acc = 100.0 * correct5 / total
        print(f"Evaluated {total} samples — Top1: {top1_acc:.2f}%, Top5: {top5_acc:.2f}%")
