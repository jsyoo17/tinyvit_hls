# src/test_utils.py

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import time
import numpy as np
import timm

def load_tinyvit_5m_local():
    """
    Load TinyViT 5M 224x224 pretrained on ImageNet-1k from timm,
    saving a local copy of the state_dict and full model in ../models/
    Returns:
        model: TinyViT model
        device: torch.device used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------------------------------
    # 1. Models directory
    # -------------------------------
    models_dir = Path("../models")
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
    state_fname = "tinyvit_5m_224_in1k_state_dict.pth"
    full_fname  = "tinyvit_5m_224_in1k_fullmodel.pth"

    state_path = models_dir / state_fname
    full_path  = models_dir / full_fname

    # -------------------------------
    # 4. Load TinyViT
    # -------------------------------

    model_name = "tiny_vit_5m_224.in1k"

    if state_path.exists():
        print(f"Found saved TinyViT state_dict at {state_path.resolve()}")
        print("Building TinyViT architecture without pretrained weights...")
        model = timm.create_model(model_name, pretrained=False)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        print("Downloading / using TinyViT pretrained weights (via HF HUB)...")
        model = timm.create_model(model_name, pretrained=True)

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

    model.eval()
    return model

def create_imagenet_subset_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    calib_per_class: int = 1,
    max_calib_images: Optional[int] = None,
    max_eval_images: Optional[int] = None,
    img_size: int = 224,
    resize_shorter: int = 256,
    shuffle_eval: bool = False,
    pin_memory: bool = True,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, Subset, Subset, datasets.ImageFolder]:
    """
    Create calibration and evaluation DataLoaders from an ImageNet-style subset.

    Splits the dataset into:
      - calibration set: first `calib_per_class` images per class
      - eval set: all remaining images

    Args:
        root: Root folder of the subset (ImageFolder layout).
        batch_size: Batch size for both loaders.
        num_workers: DataLoader workers.
        calib_per_class: How many images per class to use for calibration.
        max_calib_images: Optional cap on total number of calib images.
        max_eval_images: Optional cap on total number of eval images.
        img_size: Final crop size (e.g., 224).
        resize_shorter: Resize shorter side before center crop (e.g., 256).
        shuffle_eval: Whether to shuffle eval loader.
        pin_memory: DataLoader pin_memory flag.
        verbose: Print summary info.

    Returns:
        calib_loader, eval_loader, calib_ds, eval_ds, full_dataset
    """

    if not os.path.isdir(root):
        raise FileNotFoundError(f"ImageNet subset root not found: {root}")

    # Standard ImageNet transform
    imagenet_transform = transforms.Compose([
        transforms.Resize(resize_shorter),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Full dataset
    dataset = datasets.ImageFolder(root=root, transform=imagenet_transform)

    # Build calibration indices: first `calib_per_class` indices per class
    per_class_count = {}
    calib_idx = []

    # dataset.samples is a list of (path, label)
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in per_class_count:
            per_class_count[label] = 0

        if per_class_count[label] < calib_per_class:
            calib_idx.append(idx)
            per_class_count[label] += 1

    # Optional cap on total calibration images
    if max_calib_images is not None:
        calib_idx = calib_idx[:max_calib_images]

    calib_set = set(calib_idx)

    # Eval indices = all others
    eval_idx = [i for i in range(len(dataset)) if i not in calib_set]

    # Optional cap on eval images
    if max_eval_images is not None:
        eval_idx = eval_idx[:max_eval_images]

    calib_ds = Subset(dataset, calib_idx)
    eval_ds = Subset(dataset, eval_idx)

    calib_loader = DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=shuffle_eval,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if verbose:
        print(f"[create_imagenet_subset_loaders] Root: {root}")
        print(f"  Total images:     {len(dataset)}")
        print(f"  Classes:          {len(dataset.classes)}")
        print(f"  Calib per class:  {calib_per_class}")
        print(f"  Calibration set:  {len(calib_ds)} images")
        print(f"  Evaluation set:   {len(eval_ds)} images")
        print(f"  Batch size:       {batch_size}")

    return calib_loader, eval_loader, calib_ds, eval_ds, dataset

def save_intermediate_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    modules: Dict[str, torch.nn.Module],
    save_path: str,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
):
    """
    Run `model` on `dataloader` and save intermediate outputs from given modules.

    Args:
        model:        PyTorch model.
        dataloader:   DataLoader providing (inputs, labels) or (inputs, ...).
                      Labels are ignored.
        modules:      Dict of {name: module} whose forward outputs you want to capture.
                      Example: {"patch_conv1": model.patch_embed.seq[0].c}
        save_path:    Path to save a .pt file (torch.save).
        device:       Device for model and inputs ("cuda" or "cpu").
        max_batches:  Optionally limit number of batches processed.
        dtype:        Input dtype for the model (e.g. torch.float32).
        verbose:      Print progress info.

    Saved file (torch.save) has the structure:
        {
          "intermediates": {
              name1: Tensor [N, ...],  # concatenated over all batches
              name2: Tensor [N, ...],
              ...
          },
          "meta": {
              "num_samples": int,
              "batch_size": int,
              "module_names": List[str],
              "device": str,
              "dtype": str,
              "model_class": str,
          }
        }
    """
    model_was_training = model.training
    model.eval()
    model.to(device)

    # Storage for outputs per module
    buffer = {name: [] for name in modules.keys()}

    # Register forward hooks
    hooks = []

    def make_hook(name):
        def hook_fn(module, inputs, output):
            # Store detached CPU tensor
            buffer[name].append(output.detach().cpu())
        return hook_fn

    for name, module in modules.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    if verbose:
        print(f"[save_intermediate_outputs] Saving to: {save_path}")
        print(f"  Capturing modules: {list(modules.keys())}")

    num_samples = 0
    batch_size_seen = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # batch could be (images, labels) or (images, ...)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device=device, dtype=dtype)
            if batch_size_seen is None:
                batch_size_seen = images.size(0)

            # Forward pass (hooks will capture intermediates)
            _ = model(images)

            num_samples += images.size(0)
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches, {num_samples} samples")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate outputs for each module
    intermediates = {}
    for name, tensors in buffer.items():
        if len(tensors) == 0:
            if verbose:
                print(f"  Warning: no outputs captured for module '{name}'")
            continue
        intermediates[name] = torch.cat(tensors, dim=0)

    if verbose:
        for name, t in intermediates.items():
            print(f"  '{name}': saved tensor shape {t.shape}")

    meta = {
        "num_samples": num_samples,
        "batch_size": batch_size_seen,
        "module_names": list(intermediates.keys()),
        "device": device,
        "dtype": str(dtype),
        "model_class": model.__class__.__name__,
    }

    payload = {
        "intermediates": intermediates,
        "meta": meta,
    }

    torch.save(payload, save_path)

    if model_was_training:
        model.train()

    if verbose:
        print("[save_intermediate_outputs] Done.")

def compare_intermediate_outputs(
    ref_path: str,
    test_path: str,
    key_map: Optional[Dict[str, str]] = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    verbose: bool = True,
):
    """
    Compare intermediate outputs saved by `save_intermediate_outputs`.

    Args:
        ref_path:  Path to reference .pt file (e.g., float32 model).
        test_path: Path to test .pt file (e.g., fixed-point model).
        key_map:   Optional dict mapping ref_name -> test_name.
                   If None, assume both files share the same keys.
        atol:      Absolute tolerance (for reporting whether close).
        rtol:      Relative tolerance.
        verbose:   Whether to print a summary.

    Returns:
        results: dict keyed by ref tensor name:
            {
              "tensor_name": {
                  "ref_name": str,
                  "test_name": str,
                  "mse": float,
                  "mae": float,
                  "max_abs": float,
                  "shape": tuple,
                  "close_fraction": float,  # fraction of elements within atol+rtol*|ref|
              },
              ...
            }
    """
    ref = torch.load(ref_path, map_location="cpu")
    test = torch.load(test_path, map_location="cpu")

    ref_int = ref.get("intermediates", {})
    test_int = test.get("intermediates", {})

    if verbose:
        print(f"[compare_intermediate_outputs]")
        print(f"  Ref file:  {ref_path}")
        print(f"  Test file: {test_path}")
        print(f"  Ref keys:  {list(ref_int.keys())}")
        print(f"  Test keys: {list(test_int.keys())}")

    results = {}

    for ref_name, ref_tensor in ref_int.items():
        # Determine corresponding test name
        if key_map is not None:
            test_name = key_map.get(ref_name, None)
            if test_name is None:
                if verbose:
                    print(f"  [WARN] No mapping for ref key '{ref_name}' in key_map, skipping.")
                continue
        else:
            test_name = ref_name

        if test_name not in test_int:
            if verbose:
                print(f"  [WARN] Test intermediates missing key '{test_name}', skipping.")
            continue

        test_tensor = test_int[test_name]

        if ref_tensor.shape != test_tensor.shape:
            if verbose:
                print(f"  [WARN] Shape mismatch for '{ref_name}' vs '{test_name}': "
                      f"{ref_tensor.shape} vs {test_tensor.shape}, skipping.")
            continue

        ref_flat = ref_tensor.reshape(-1)
        test_flat = test_tensor.reshape(-1)

        diff = test_flat - ref_flat
        mse = (diff.pow(2).mean()).item()
        mae = diff.abs().mean().item()
        max_abs = diff.abs().max().item()

        # fraction of elements within atol + rtol * |ref|
        tol = atol + rtol * ref_flat.abs()
        close = (diff.abs() <= tol).float().mean().item()

        stats = {
            "ref_name": ref_name,
            "test_name": test_name,
            "mse": mse,
            "mae": mae,
            "max_abs": max_abs,
            "shape": tuple(ref_tensor.shape),
            "close_fraction": close,
        }
        results[ref_name] = stats

    if verbose:
        print("\n[Comparison results]")
        for name, s in results.items():
            print(
                f"  {name}: shape={s['shape']}, "
                f"MSE={s['mse']:.6e}, MAE={s['mae']:.6e}, "
                f"MaxAbs={s['max_abs']:.6e}, close_frac={s['close_fraction']:.4f}"
            )

    return results
