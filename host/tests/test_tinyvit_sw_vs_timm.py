#!/usr/bin/env python3
"""
test_tinyvit_sw_vs_timm.py

Compare TinyVit5MSW (software reference) vs timm TinyViT-5M
for different parts of the network:

  - patch_embed: timm.patch_embed(x) vs sw.patch_embed(x)
  - stage0    : timm.stages[0](u) vs sw.stages[0](u)
  - stage1    : timm.stages[1](v) vs sw.stages[1](v), with same input v
  - features  : patch_embed -> stage0 -> stage1 vs sw.forward_features(x)

Default input:
  Uses real tensors from:
    data/test_vectors/input_50/input_50.bin

You can override to use random input with --use-random.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import timm


# ---------------------------------------------------------------------
# Helper: set up paths / imports
# ---------------------------------------------------------------------

def setup_imports_and_paths():
    """Ensure we can import host/tinyvit and set models_dir."""
    here = Path(__file__).resolve()
    proj_root = here.parents[2]  # host/tests/ -> host/ -> proj_root
    host_dir = proj_root / "host"
    models_dir = proj_root / "data" / "models"

    if str(host_dir) not in sys.path:
        sys.path.insert(0, str(host_dir))

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir.resolve())
    os.environ["HF_HUB_CACHE"] = str(models_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(models_dir.resolve())

    models_dir.mkdir(parents=True, exist_ok=True)

    return proj_root, host_dir, models_dir


# ---------------------------------------------------------------------
# Helper: load TinyViT reference model
# ---------------------------------------------------------------------

def load_timm_tinyvit(model_name: str, models_dir: Path):
    """Load timm TinyViT (using cached state if exists)."""
    safe_name = model_name.replace(".", "_")
    state_path = models_dir / f"{safe_name}_state_dict.pth"

    if state_path.exists():
        print(f"[TEST] Using cached state_dict at {state_path}")
        model_ref = timm.create_model(model_name, pretrained=False)
        state = torch.load(state_path, map_location="cpu")
        incompatible = model_ref.load_state_dict(state, strict=False)
        print(f"[TEST] timm load_state_dict(strict=False): "
              f"missing={len(incompatible.missing_keys)}, "
              f"unexpected={len(incompatible.unexpected_keys)}")
    else:
        print(f"[TEST] Downloading pretrained TinyViT ({model_name})...")
        model_ref = timm.create_model(model_name, pretrained=True)
        torch.save(model_ref.state_dict(), state_path)
        print(f"[TEST] Saved state_dict to {state_path}")

    model_ref.eval()
    return model_ref


# ---------------------------------------------------------------------
# Helper: load packed ImageNet inputs (input_N.bin)
# ---------------------------------------------------------------------

def load_input_bin(num_images: int, input_root: Path) -> torch.Tensor:
    """
    Load packed input_N.bin:

      int32 N, C, H, W
      float32 data[N*C*H*W]

    Returns: torch.Tensor [N, C, H, W]
    """
    input_dir = input_root / f"input_{num_images}"
    input_path = input_dir / f"input_{num_images}.bin"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[TEST] Loading input tensor from {input_path}")
    import numpy as np
    with open(input_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header.size != 4:
            raise RuntimeError(f"Failed to read header from {input_path}")
        N, C, H, W = header.tolist()
        data = np.fromfile(f, dtype=np.float32)

    expected = N * C * H * W
    if data.size != expected:
        raise RuntimeError(
            f"Data size mismatch in {input_path}: got {data.size}, expected {expected}"
        )

    arr = data.reshape(N, C, H, W)
    x = torch.from_numpy(arr)  # float32
    print(f"[TEST] Loaded input tensor shape: {tuple(x.shape)}")
    return x


def get_test_input(
    device: torch.device,
    batch_size: int,
    num_images: int,
    input_root: Path,
    use_random: bool,
) -> torch.Tensor:
    """
    Get test input tensor [B, 3, 224, 224].

    - Default: load data/test_vectors/input_<num_images>/input_<num_images>.bin
      and take the first B samples.
    - If --use-random is set, or file missing, uses random instead.
    """
    if use_random:
        print("[TEST] Using RANDOM input (override with --use-random).")
        torch.manual_seed(0)
        return torch.randn(batch_size, 3, 224, 224, device=device)

    try:
        x_all = load_input_bin(num_images, input_root)  # [N, C, H, W]
    except FileNotFoundError as e:
        print(f"[TEST][WARN] {e}")
        print("[TEST][WARN] Falling back to random input.")
        torch.manual_seed(0)
        return torch.randn(batch_size, 3, 224, 224, device=device)

    if x_all.size(0) < batch_size:
        print(f"[TEST][WARN] input has only {x_all.size(0)} images; "
              f"reducing batch_size from {batch_size} to {x_all.size(0)}.")
        batch_size = x_all.size(0)

    x = x_all[:batch_size].to(device)
    return x


# ---------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------

def _report_diff(name: str, diff: torch.Tensor, tol: float = 1e-6):
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"[TEST] {name} max_abs_diff  = {max_abs:.8g}")
    print(f"[TEST] {name} mean_abs_diff = {mean_abs:.8g}")
    if max_abs <= tol:
        print(f"[TEST] {name} PASS: max_abs_diff <= {tol}")
    else:
        print(f"[TEST] {name} WARNING: max_abs_diff > {tol} (={max_abs:.8g})")


def run_test_patch_embed(model_ref, model_sw, x):
    print("\n[TEST] === patch_embed ===")

    with torch.no_grad():
        y_ref = model_ref.patch_embed(x)       # [B, 64, 56, 56]
        y_sw  = model_sw.patch_embed(x)        # [B, 64, 56, 56]

    print("[TEST] y_ref shape:", tuple(y_ref.shape))
    print("[TEST] y_sw  shape:", tuple(y_sw.shape))

    diff = (y_ref - y_sw).abs()
    _report_diff("patch_embed", diff)


def run_test_stage0(model_ref, model_sw, x):
    """
    We feed the SAME input u to both stage0 implementations to isolate that block:
      u = model_ref.patch_embed(x)
      y_ref = model_ref.stages[0](u)
      y_sw  = model_sw.stages[0](u)
    """
    print("\n[TEST] === stage0 (ConvLayer / MBConv stack) ===")

    with torch.no_grad():
        u = model_ref.patch_embed(x)       # [B, 64, 56, 56]
        y_ref = model_ref.stages[0](u)     # [B, 64, 56, 56]
        y_sw  = model_sw.stages[0](u)      # [B, 64, 56, 56]

    print("[TEST] u     shape:", tuple(u.shape))
    print("[TEST] y_ref shape:", tuple(y_ref.shape))
    print("[TEST] y_sw  shape:", tuple(y_sw.shape))

    diff = (y_ref - y_sw).abs()
    _report_diff("stage0", diff)


def run_test_stage1(model_ref, model_sw, x):
    """
    Isolated test of stage1 TinyVitStage:

      v = model_ref.stages[0](patch_embed(x))
      y_ref = model_ref.stages[1](v)
      y_sw  = model_sw.stages[1](v)
    """
    print("\n[TEST] === stage1 (TinyVitStage / attention) ===")

    with torch.no_grad():
        u = model_ref.patch_embed(x)
        v = model_ref.stages[0](u)       # common stage0 output
        y_ref = model_ref.stages[1](v)   # [B, 128, 28, 28]
        y_sw  = model_sw.stages[1](v)    # [B, 128, 28, 28]

    print("[TEST] v     shape:", tuple(v.shape))
    print("[TEST] y_ref shape:", tuple(y_ref.shape))
    print("[TEST] y_sw  shape:", tuple(y_sw.shape))

    diff = (y_ref - y_sw).abs()
    _report_diff("stage1", diff)


def run_test_features(model_ref, model_sw, x):
    """
    Compare full forward_features up to stage1:

      ref: x -> patch_embed -> stages[0] -> stages[1]
      sw : x -> forward_features(x)
    """
    print("\n[TEST] === forward_features (through stage1) ===")

    with torch.no_grad():
        u0 = model_ref.patch_embed(x)
        u1 = model_ref.stages[0](u0)
        y_ref = model_ref.stages[1](u1)     # [B, 128, 28, 28]

        y_sw = model_sw.forward_features(x) # same shape

    print("[TEST] y_ref shape:", tuple(y_ref.shape))
    print("[TEST] y_sw  shape:", tuple(y_sw.shape))

    diff = (y_ref - y_sw).abs()
    _report_diff("features", diff)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare TinyVit5MSW vs timm TinyViT-5M (patch_embed, stage0, stage1, features)."
    )
    parser.add_argument(
        "--tests",
        type=str,
        nargs="+",
        default=["all"],
        help="Which tests to run: patch_embed, stage0, stage1, features, all",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for test inputs.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="N for input_N.bin (default: 50 → data/test_vectors/input_50/input_50.bin)",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/test_vectors",
        help="Root directory containing input_N/ subfolders.",
    )
    parser.add_argument(
        "--use-random",
        action="store_true",
        help="Use random input instead of input_N.bin.",
    )
    args = parser.parse_args()

    tests = set([t.lower() for t in args.tests])
    if "all" in tests:
        tests = {"patch_embed", "stage0", "stage1", "features"}

    proj_root, host_dir, models_dir = setup_imports_and_paths()
    from tinyvit.tiny_vit_sw import create_tiny_vit_5m_sw

    model_name = "tiny_vit_5m_224.dist_in22k_ft_in1k"
    device = torch.device("cpu")  # keep CPU for exact comparability

    print("[TEST] Project root:", proj_root)
    print("[TEST] Using device:", device)

    # Build models
    model_ref = load_timm_tinyvit(model_name, models_dir)
    model_sw = create_tiny_vit_5m_sw(
        model_name=model_name,
        models_dir=str(models_dir),
        device=str(device),
    )

    # Ensure both are in inference mode (important for BatchNorm & LayerNorm!)
    model_ref.eval()
    model_sw.eval()

    # Prepare input (real images by default)
    input_root = proj_root / args.input_root
    x = get_test_input(
        device=device,
        batch_size=args.batch_size,
        num_images=args.num_images,
        input_root=input_root,
        use_random=args.use_random,
    )

    # Run selected tests
    if "patch_embed" in tests:
        run_test_patch_embed(model_ref, model_sw, x)
    if "stage0" in tests:
        run_test_stage0(model_ref, model_sw, x)
    if "stage1" in tests:
        run_test_stage1(model_ref, model_sw, x)
    if "features" in tests:
        run_test_features(model_ref, model_sw, x)


if __name__ == "__main__":
    main()
