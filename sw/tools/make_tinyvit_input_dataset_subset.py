#!/usr/bin/env python3
"""
make_tinyvit_input_dataset_subset.py

example usage in powershell:
    python sw/tools/make_tinyvit_input_dataset_subset.py `
    --data-root data/datasets/imagenet1k_val_subset/ILSVRC2012_img_val_subset `
    --num-images 1000

Creates a folder:
    data/test_vectors/input_<N>/

Inside the folder:
    input_<N>.bin
    labels_<N>_golden.bin

Binary formats:
  input_<N>.bin:
    int32 N, C, H, W
    float32 data[N * C * H * W]

  labels_<N>_golden.bin:
    int32 N
    int32 labels[N]
"""

import os
import argparse
from typing import List

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def list_folders(path):
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sorted(folders)


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    ]
    return sorted(files)


def save_input_bin(path, tensor):
    t = tensor.detach().cpu().contiguous()
    assert t.ndim == 4
    N, C, H, W = t.shape

    header = np.array([N, C, H, W], dtype=np.int32)
    data = t.numpy().astype(np.float32).reshape(-1)

    with open(path, "wb") as f:
        header.tofile(f)
        data.tofile(f)


def save_labels_bin(path, labels):
    labels = np.array(labels, dtype=np.int32)
    N = len(labels)
    header = np.array([N], dtype=np.int32)
    with open(path, "wb") as f:
        header.tofile(f)
        labels.tofile(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=str,
                        help="Folder containing class folders (000, 001, ...)")
    parser.add_argument("--num-images", default=1000, type=int)
    parser.add_argument("--img-size", default=224, type=int)
    args = parser.parse_args()

    data_root = args.data_root
    num_images_req = args.num_images

    # ---------------------------------------
    # Output folder: data/test_vectors/input_<N>
    # ---------------------------------------
    out_dir = f"data/test_vectors/input_{num_images_req}"
    os.makedirs(out_dir, exist_ok=True)

    # Save paths
    input_bin = os.path.join(out_dir, f"input_{num_images_req}.bin")
    labels_bin = os.path.join(out_dir, f"labels_{num_images_req}_golden.bin")

    # ---------------------------------------
    # Collect class folders
    # ---------------------------------------
    class_folders = list_folders(data_root)
    num_classes = len(class_folders)
    print(f"[INFO] Found {num_classes} class folders.")

    base = num_images_req // num_classes
    extra = num_images_req % num_classes
    print(f"[INFO] base={base}, extra={extra}")

    transform = build_transform(args.img_size)

    all_imgs = []
    all_labels = []
    total = 0

    for class_idx, folder in enumerate(class_folders):
        want = base + (1 if class_idx < extra else 0)
        if want == 0:
            continue

        class_dir = os.path.join(data_root, folder)
        img_files = list_images(class_dir)

        take = min(want, len(img_files))

        for i in range(take):
            img_path = os.path.join(class_dir, img_files[i])
            with Image.open(img_path).convert("RGB") as img:
                x = transform(img)  # [C,H,W]
            all_imgs.append(x.unsqueeze(0))
            all_labels.append(class_idx)
            total += 1

            if total >= num_images_req:
                break

        if total >= num_images_req:
            break

    all_imgs = torch.cat(all_imgs, dim=0)
    print(f"[INFO] Final shape = {all_imgs.shape}")

    # Save files
    save_input_bin(input_bin, all_imgs)
    save_labels_bin(labels_bin, all_labels)

    print(f"[OK] Created dataset:")
    print(f"     {input_bin}")
    print(f"     {labels_bin}")


if __name__ == "__main__":
    main()
