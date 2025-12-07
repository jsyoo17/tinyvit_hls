#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import timm


"""
example usage:
    # single module
    python host/tools/dump_tinyvit_activation.py `
      --num-images 1000 `
      --module patch_embed.conv1.conv `
      --device cpu

    # multiple modules
    python host/tools/dump_tinyvit_activation.py `
      --num-images 100 `
      --modules patch_embed.conv1.conv stages.0.blocks.0.local_conv `
      --device cpu

    # all modules (ALL tensor outputs with ndim >= 1 will be saved, with size prompt)
    python host/tools/dump_tinyvit_activation.py `
      --num-images 50 `
      --all `
      --device cpu


Inputs (input_N.bin):
    Format (4D only, for model input):
        int32 N, C, H, W
        float32 data[N*C*H*W]

Activations (this script):
    data/test_vectors/input_N/<module_name_with_underscores>_output_N_golden.bin

    Format (ND, generic):
        int32 ndim
        int32 dim_0, ..., dim_{ndim-1}
        float32 data[∏ dim_i]

Plus JSON manifest:
    data/test_vectors/input_N/activations_manifest_N.json
"""


# ----------------------------
# Binary tensor I/O helpers
# ----------------------------

def load_tensor_bin(path: str) -> torch.Tensor:
    """
    Load [N, C, H, W] tensor from .bin file with format:
        int32 N, C, H, W
        float32 data[N*C*H*W] (row-major)

    This is used for input_N.bin only.
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header.size != 4:
            raise RuntimeError(f"Failed to read header from {path}")
        N, C, H, W = header.tolist()
        data = np.fromfile(f, dtype=np.float32)

    expected = N * C * H * W
    if data.size != expected:
        raise RuntimeError(
            f"Data size mismatch in {path}: got {data.size}, expected {expected}"
        )

    arr = data.reshape(N, C, H, W)
    return torch.from_numpy(arr)


def save_activation_bin(path: str, tensor: torch.Tensor) -> None:
    """
    Save an activation tensor of arbitrary shape (e.g. 1D, 2D, 3D, 4D)
    in ND .bin format:

        int32 ndim
        int32 dim_0, ..., dim_{ndim-1}
        float32 data[∏ dim_i]
    """
    t = tensor.detach().cpu().contiguous()
    ndim = t.ndim
    if ndim < 1:
        raise ValueError(f"Expected ndim >= 1, got {ndim} for tensor {t.shape}")
    shape = list(t.shape)

    header = np.array([ndim] + shape, dtype=np.int32)
    data = t.numpy().astype(np.float32).ravel()

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "wb") as f:
        header.tofile(f)
        data.tofile(f)

    print(f"[INFO] Saved activation {tuple(shape)} (ndim={ndim}) to {path}")


# ----------------------------
# Model loading helper
# ----------------------------

def load_tinyvit_model(
    model_name: str,
    models_dir: Path,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Load TinyViT model using timm with HF cache redirected to models_dir.

    If a state_dict file already exists in models_dir, use it.
    Otherwise, download pretrained weights and save state_dict for future runs.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir = models_dir.resolve()

    print("[INFO] Using models directory:", models_dir)

    # Redirect HF + timm caches
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    os.environ["XDG_CACHE_HOME"] = str(models_dir)

    safe_name = model_name.replace(".", "_")
    state_path = models_dir / f"{safe_name}_state_dict.pth"

    if state_path.exists():
        print(f"[INFO] Found saved TinyViT state_dict at {state_path}")
        print("[INFO] Building TinyViT architecture without pretrained=True...")
        model = timm.create_model(model_name, pretrained=False)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        print("[INFO] Downloading TinyViT pretrained weights via timm/HF...")
        model = timm.create_model(model_name, pretrained=True)
        try:
            torch.save(model.state_dict(), state_path)
            print(f"[INFO] Saved state_dict to {state_path}")
        except Exception as e:
            print(f"[WARN] Failed to save state_dict: {e}")

    model.to(device)
    model.eval()
    return model


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dump TinyViT activations at specified module paths "
                    "starting from packed input_N.bin."
    )

    parser.add_argument(
        "--num-images",
        type=int,
        required=True,
        help="N used in input_N.bin (e.g. 1000 for data/test_vectors/input_1000/input_1000.bin)",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Single dotted module path inside TinyViT, e.g. 'patch_embed.conv1.conv'",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="*",
        default=None,
        help="Multiple module paths, e.g. 'patch_embed.conv1.conv stages.0.blocks.0.local_conv'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Dump activations for all modules (ALL tensor outputs with ndim >= 1).",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/test_vectors",
        help="Root directory containing input_N/ subfolders",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tiny_vit_5m_224.dist_in22k_ft_in1k",
        help="timm TinyViT model name",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory to store TinyViT weights/cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run model on (e.g. 'cpu' or 'cuda')",
    )

    args = parser.parse_args()

    N = args.num_images
    input_dir = Path(args.input_root) / f"input_{N}"
    input_path = input_dir / f"input_{N}.bin"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Loading input tensor from {input_path}")
    x = load_tensor_bin(str(input_path))  # [N, 3, H, W]
    print(f"[INFO] Input shape: {tuple(x.shape)}")

    device = torch.device(args.device)

    # Load model
    model = load_tinyvit_model(
        model_name=args.model_name,
        models_dir=Path(args.models_dir),
        device=device,
    )

    # Decide which module names to hook
    if args.all:
        module_names = [name for name, _ in model.named_modules() if name]
        print(f"[INFO] --all selected, will hook {len(module_names)} modules.")
    else:
        module_names = []
        if args.module is not None:
            module_names.append(args.module)
        if args.modules is not None:
            module_names.extend(args.modules)

        if not module_names:
            raise ValueError(
                "No modules specified. Use --module, --modules, or --all."
            )

        print(f"[INFO] Will hook {len(module_names)} module(s):")
        for name in module_names:
            print(f"   - {name}")

    name_to_module = dict(model.named_modules())

    # Register hooks
    captured = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            captured[name] = out.detach()
        return hook_fn

    handles = []
    for name in module_names:
        if name not in name_to_module:
            raise KeyError(f"Module name '{name}' not found in model.named_modules().")
        m = name_to_module[name]
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    # Forward pass
    with torch.no_grad():
        _ = model(x.to(device))

    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError(
            "No outputs captured. Check module names or whether modules are used in forward."
        )

    # Include ALL tensors with ndim >= 1, estimate total size
    tensors_to_save = {}
    total_bytes = 0
    for name, t in captured.items():
        t_cpu = t.detach().cpu()
        if t_cpu.ndim < 1:
            print(f"[WARN] Skipping '{name}' with ndim=0 (scalar) shape={tuple(t_cpu.shape)}")
            continue

        num_elems = int(np.prod(t_cpu.shape))
        header_ints = 1 + t_cpu.ndim  # ndim + dims
        bytes_this = header_ints * 4 + num_elems * 4  # int32 + float32

        total_bytes += bytes_this
        tensors_to_save[name] = t_cpu

    if not tensors_to_save:
        print("[WARN] No tensor outputs (ndim >= 1) to save. Nothing to do.")
        return

    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    print(f"[INFO] Will save {len(tensors_to_save)} activation tensor(s).")
    print(f"[INFO] Estimated total size: {total_mb:.2f} MB ({total_gb:.3f} GB).")

    resp = input("Proceed with saving these activations? [y/N]: ").strip().lower()
    if resp not in ("y", "yes"):
        print("[INFO] Aborting without saving activations.")
        return

    # Save tensors + collect manifest info
    manifest = {
        "num_images": N,
        "model_name": args.model_name,
        "input_file": str(input_path),
        "dtype": "float32",
        "activation_bin_format": "ndim:int32, dims:int32[ndim], data:float32[∏dims]",
        "tensors": [],
    }

    for name, t_cpu in tensors_to_save.items():
        module_name_flat = name.replace(".", "_")
        out_fname = f"{module_name_flat}_output_{N}_golden.bin"
        out_path = input_dir / out_fname
        print(f"[INFO] Saving activation for '{name}' -> {out_fname}")
        save_activation_bin(str(out_path), t_cpu)

        entry = {
            "module": name,
            "file": out_fname,
            "shape": list(t_cpu.shape),
            "ndim": int(t_cpu.ndim),
            "dtype": "float32",
        }
        manifest["tensors"].append(entry)

    # Save manifest JSON
    manifest_path = input_dir / f"activations_manifest_{N}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved activation manifest to {manifest_path}")

    print("[DONE] Activation dump complete.")


if __name__ == "__main__":
    main()
