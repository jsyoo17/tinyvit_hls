#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from functools import reduce
import operator

import numpy as np
import torch
import timm


# ----------------------------
# Binary tensor I/O helpers

"""
example usage:
    python sw/tests/dump_tinyvit_activation.py `
    --num-images 1000 `
    --module patch_embed.conv1.conv `
    --device cpu
    
this will produce:
    data/test_vectors/input_1000/patch_embed_conv1_conv_output_1000_golden.bin
"""  

def load_tensor_bin(path: str) -> torch.Tensor:
    """
    Load [N, C, H, W] tensor from .bin file with format:
        int32 N, C, H, W
        float32 data[N*C*H*W] (row-major)
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


def save_tensor_bin(path: str, tensor: torch.Tensor) -> None:
    """
    Save [N, C, H, W] tensor in .bin format:
        int32 N, C, H, W
        float32 data[N*C*H*W]
    """
    t = tensor.detach().cpu().contiguous()
    if t.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got shape {t.shape}")
    N, C, H, W = t.shape

    header = np.array([N, C, H, W], dtype=np.int32)
    data = t.numpy().astype(np.float32).ravel()

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "wb") as f:
        header.tofile(f)
        data.tofile(f)

    print(f"[INFO] Saved tensor {t.shape} to {path}")


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

    # Redirect HF + timm caches (same idea as your notebook snippet)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    os.environ["XDG_CACHE_HOME"] = str(models_dir)

    # Derive a safe filename for the state_dict
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
# Module lookup helper
# ----------------------------

def get_module_by_path(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    """
    Resolve a dotted module path like "patch_embed.conv1.conv"
    into the actual submodule.
    """
    parts = module_path.split(".")
    try:
        module = reduce(getattr, parts, model)
    except AttributeError as e:
        raise AttributeError(f"Could not resolve module path '{module_path}': {e}")
    return module


# ----------------------------
# Main script
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dump TinyViT activation at a specified module path "
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
        required=True,
        help="Dotted module path inside TinyViT, e.g. 'patch_embed.conv1.conv'",
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

    # Resolve target module
    target_module_path = args.module
    target_module = get_module_by_path(model, target_module_path)
    print(f"[INFO] Hooking module: {target_module_path} -> {target_module}")

    # Register forward hook to capture output
    captured = {}

    def hook_fn(module, inp, out):
        # Save the output tensor (before moving to CPU)
        captured["output"] = out.detach()

    handle = target_module.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        y_model = model(x.to(device))

    handle.remove()

    if "output" not in captured:
        raise RuntimeError(
            f"No output captured from module '{target_module_path}'. "
            "Was the module actually used during the forward pass?"
        )

    out_tensor = captured["output"].detach().cpu()
    print(f"[INFO] Captured activation shape from '{target_module_path}': {tuple(out_tensor.shape)}")

    if out_tensor.ndim != 4:
        raise ValueError(
            f"Captured activation is not 4D (got {out_tensor.shape}). "
            "Current .bin format assumes [N,C,H,W]."
        )

    # Build output filename: replace '.' by '_' and append _output_<N>.bin
    module_name_flat = target_module_path.replace(".", "_")
    out_fname = f"{module_name_flat}_output_{N}_golden.bin"
    out_path = input_dir / out_fname

    save_tensor_bin(str(out_path), out_tensor)
    print("[DONE] Activation dump complete.")


if __name__ == "__main__":
    main()
