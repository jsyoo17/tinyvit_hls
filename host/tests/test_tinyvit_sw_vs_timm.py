# host/tests/test_tinyvit_sw_vs_timm.py

#!/usr/bin/env python3
"""
test_tinyvit_sw_vs_timm.py

Compare timm TinyViT-5M (tiny_vit_5m_224.dist_in22k_ft_in1k) vs TinyVitSWSmall
for different parts of the forward path.

Default input: real ImageNet subset packed into:
    data/test_vectors/input_50/input_50.bin

Fallback: random Gaussian images if the .bin file is missing.

Usage:
    python host/tests/test_tinyvit_sw_vs_timm.py --tests all
    python host/tests/test_tinyvit_sw_vs_timm.py --tests patch_embed stage0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from timm import create_model as timm_create_model

# make host/ a package root so we can import tinyvit.tiny_vit_sw
THIS_FILE = Path(__file__).resolve()
HOST_DIR = THIS_FILE.parents[1]
PROJ_ROOT = HOST_DIR.parent
if str(HOST_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(HOST_DIR))

from tinyvit.tiny_vit_sw import TinyVitSWSmall, create_tiny_vit_5m_sw  # type: ignore


# -------------------------------------------------------------------------
# Binary tensor loader compatible with make_tinyvit_input_dataset_subset.py
# -------------------------------------------------------------------------

def load_tensor_bin(path: Path) -> torch.Tensor:
    """
    Load [N, C, H, W] tensor from .bin file with format:
        int32 N, C, H, W
        float32 data[N*C*H*W]
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


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def compare_tensors(name: str, y_ref: torch.Tensor, y_sw: torch.Tensor) -> Dict[str, Any]:
    assert y_ref.shape == y_sw.shape, (
        f"Shape mismatch for {name}: ref {tuple(y_ref.shape)} vs sw {tuple(y_sw.shape)}"
    )
    diff = (y_ref - y_sw).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[TEST] {name} y_ref shape: {tuple(y_ref.shape)}")
    print(f"[TEST] {name} y_sw  shape: {tuple(y_sw.shape)}")
    print(f"[TEST] {name} max_abs_diff  = {max_diff}")
    print(f"[TEST] {name} mean_abs_diff = {mean_diff}")
    tol = 1e-5
    if max_diff > tol:
        print(f"[TEST] {name} WARNING: max_abs_diff > {tol} (= {max_diff})")
    else:
        print(f"[TEST] {name} OK: within tolerance {tol}")
    print()
    return {
        "name": name,
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "ok": bool(max_diff <= tol),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help="Which tests to run: patch_embed, stage0, stage1, stage2, stage3, features, logits, or 'all'.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="N for input_N.bin under data/test_vectors/ (default: 50).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on, e.g. 'cpu' or 'cuda'",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(">>")
    print(f"[TEST] Project root: {PROJ_ROOT}")
    print(f"[TEST] Using device: {device}")

    # models directory
    models_dir = PROJ_ROOT / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load pretrained timm TinyViT
    # ------------------------------------------------------------------
    state_fname = "tiny_vit_5m_224_dist_in22k_ft_in1k_state_dict.pth"
    state_path = models_dir / state_fname

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir.resolve())
    os.environ["HF_HUB_CACHE"] = str(models_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(models_dir.resolve())

    if state_path.exists():
        print(f"[TEST] Using cached state_dict at {state_path}")
        state = torch.load(state_path, map_location="cpu")
        model_ref = timm_create_model(
            "tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=False
        )
        m = model_ref.load_state_dict(state, strict=False)
        print(
            f"[TEST] timm TinyViT load_state_dict(strict=False): "
            f"missing={len(m.missing_keys)}, unexpected={len(m.unexpected_keys)}"
        )
    else:
        print("[TEST] Downloading timm TinyViT pretrained weights...")
        model_ref = timm_create_model(
            "tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True
        )
        state = model_ref.state_dict()
        torch.save(state, state_path)
        print(f"[TEST] Saved state_dict to {state_path}")

    model_ref = model_ref.to(device)
    model_ref.eval()

    # ------------------------------------------------------------------
    # Load SW model with same state_dict
    # ------------------------------------------------------------------
    model_sw = TinyVitSWSmall().to(device)
    model_sw.eval()
    m2 = model_sw.load_state_dict(state, strict=False)
    print("[TEST] SW TinyViT load_state_dict(strict=False):")
    print(f"[TEST]   missing_keys   = {len(m2.missing_keys)}")
    print(f"[TEST]   unexpected_keys= {len(m2.unexpected_keys)}")
    print()

    # ------------------------------------------------------------------
    # Prepare input: real bin if exists, else random
    # ------------------------------------------------------------------
    tv_root = PROJ_ROOT / "data" / "test_vectors"
    input_dir = tv_root / f"input_{args.num_images}"
    input_path = input_dir / f"input_{args.num_images}.bin"

    if input_path.exists():
        print(f"[TEST] Loading input tensor from {input_path}")
        x = load_tensor_bin(input_path)  # [N, 3, 224, 224]
        # For speed / debugging: we can subsample to a few images
        x = x.to(device)
    else:
        print(
            f"[TEST] WARNING: {input_path} not found. "
            f"Using random Gaussian input instead."
        )
        N = args.num_images
        x = torch.randn(N, 3, 224, 224, device=device)

    # We'll use a small batch for tests to keep things fast
    if x.shape[0] > 4:
        x_test = x[:4]
        print(f"[TEST] Using first 4 images out of {x.shape[0]} for testing.")
    else:
        x_test = x
        print(f"[TEST] Using all {x.shape[0]} images for testing.")

    # ------------------------------------------------------------------
    # Actual tests
    # ------------------------------------------------------------------
    requested = set(args.tests)
    if "all" in requested:
        requested = {
            "patch_embed",
            "stage0",
            "stage1",
            "stage2",
            "stage3",
            "features",
            "logits",
        }

    results: List[Dict[str, Any]] = []

    # Helper to get intermediate outputs on timm model
    with torch.no_grad():
        # patch_embed
        x_ref_pe = model_ref.patch_embed(x_test)       # [B, 64, 56, 56]
        x_sw_pe = model_sw.patch_embed(x_test)

        if "patch_embed" in requested:
            results.append(compare_tensors("patch_embed", x_ref_pe, x_sw_pe))

        # stage0
        x_ref_s0 = model_ref.stages[0](x_ref_pe)       # [B, 64, 56, 56]
        x_sw_s0 = model_sw.stages[0](x_sw_pe)
        if "stage0" in requested:
            results.append(compare_tensors("stage0", x_ref_s0, x_sw_s0))

        # stage1
        x_ref_s1 = model_ref.stages[1](x_ref_s0)       # [B, 128, 28, 28]
        x_sw_s1 = model_sw.stages[1](x_sw_s0)
        if "stage1" in requested:
            results.append(compare_tensors("stage1", x_ref_s1, x_sw_s1))

        # stage2
        x_ref_s2 = model_ref.stages[2](x_ref_s1)       # [B, 160, 14, 14]
        x_sw_s2 = model_sw.stages[2](x_sw_s1)
        if "stage2" in requested:
            results.append(compare_tensors("stage2", x_ref_s2, x_sw_s2))

        # stage3
        x_ref_s3 = model_ref.stages[3](x_ref_s2)       # [B, 320, 7, 7]
        x_sw_s3 = model_sw.stages[3](x_sw_s2)
        if "stage3" in requested:
            results.append(compare_tensors("stage3", x_ref_s3, x_sw_s3))

        # forward_features (through all stages)
        y_ref_feats = model_ref.forward_features(x_test)
        y_sw_feats = model_sw.forward_features(x_test)
        if "features" in requested:
            results.append(compare_tensors("forward_features", y_ref_feats, y_sw_feats))

        # full logits
        logits_ref = model_ref(x_test)
        logits_sw = model_sw(x_test)
        if "logits" in requested:
            results.append(compare_tensors("logits", logits_ref, logits_sw))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("========== TinyViT SW vs timm SUMMARY ==========")
    if not results:
        print("No tests were run (check --tests argument).")
    else:
        all_ok = True
        for r in results:
            status = "OK" if r["ok"] else "FAIL"
            print(
                f"[SUMMARY] {r['name']:<18} "
                f"status={status:<4} "
                f"max_abs_diff={r['max_abs_diff']:.6e} "
                f"mean_abs_diff={r['mean_abs_diff']:.6e}"
            )
            if not r["ok"]:
                all_ok = False
        print("-----------------------------------------------")
        if all_ok:
            print("[SUMMARY] All selected tests PASSED within tolerance.")
        else:
            print("[SUMMARY] Some tests FAILED. See details above.")
    print("===============================================")


if __name__ == "__main__":
    main()
