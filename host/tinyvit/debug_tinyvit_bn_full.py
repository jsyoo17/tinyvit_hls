#!/usr/bin/env python3
"""
debug_tinyvit_bn_full.py

End-to-end check of:
  - timm TinyViT vs TinyVitSWSmall
  - BatchNorm behavior at stages.0.blocks.0.conv3.bn
  - Optional: compare against golden activation .bin

It:
  - Loads timm TinyViT with the SAME code as dump_tinyvit_activation.py
  - Loads TinyVitSWSmall with the SAME state_dict
  - Loads input_N.bin
  - Captures conv3_out, bn_out, block_out for both
  - Runs manual F.batch_norm and compares
  - If golden BN file exists, compares against that too.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm

# Import your SW model
from tiny_vit_sw import TinyVitSWSmall

# ----------------------------
# Copied helpers from dump_tinyvit_activation.py
# ----------------------------

def load_tensor_bin(path: str) -> torch.Tensor:
    """Load [N, C, H, W] tensor from .bin (int32 N,C,H,W + float32 data)."""
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


def load_tinyvit_model(
    model_name: str,
    models_dir: Path,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Same loader as dump_tinyvit_activation.py:
      - Redirect HF cache
      - Use saved state_dict if present, else download
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir = models_dir.resolve()

    print("[DBG] Using models directory:", models_dir)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    os.environ["XDG_CACHE_HOME"] = str(models_dir)

    safe_name = model_name.replace(".", "_")
    state_path = models_dir / f"{safe_name}_state_dict.pth"

    if state_path.exists():
        print(f"[DBG] Found saved TinyViT state_dict at {state_path}")
        print("[DBG] Building TinyViT architecture without pretrained=True...")
        model = timm.create_model(model_name, pretrained=False)
        state = torch.load(state_path, map_location="cpu")
        m = model.load_state_dict(state, strict=False)
        print(f"[DBG] timm.load_state_dict(strict=False): "
              f"missing={len(m.missing_keys)}, unexpected={len(m.unexpected_keys)}")
    else:
        print("[DBG] Downloading TinyViT pretrained weights via timm/HF...")
        model = timm.create_model(model_name, pretrained=True)
        state = model.state_dict()
        torch.save(state, state_path)
        print(f"[DBG] Saved state_dict to {state_path}")

    model.to(device)
    model.eval()
    return model


# ----------------------------
# Main debug logic
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deep debug of BN at stages.0.blocks.0.conv3.bn "
                    "between timm TinyViT and SW TinyVitSWSmall."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="N used in input_N.bin & golden files (default: 1000)"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/test_vectors",
        help="Root dir for input_N/ (default: data/test_vectors)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory storing TinyViT weights/cache (default: data/models)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tiny_vit_5m_224.dist_in22k_ft_in1k",
        help="timm TinyViT model name (default: tiny_vit_5m_224.dist_in22k_ft_in1k)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    N = args.num_images

    # ------------------------
    # 1) Load input_N.bin
    # ------------------------
    input_dir = Path(args.input_root) / f"input_{N}"
    input_path = input_dir / f"input_{N}.bin"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[DBG] Loading input from {input_path}")
    x_full = load_tensor_bin(str(input_path))  # [N, 3, H, W]
    print(f"[DBG] Input tensor shape: {tuple(x_full.shape)}")

    # Take just the first image to keep things light
    x = x_full[:1].to(device)
    print(f"[DBG] Using first image only: {tuple(x.shape)}")

    # ------------------------
    # 2) Build timm & SW models with shared state_dict
    # ------------------------
    print("[DBG] Building timm model (same loader as dump_tinyvit_activation.py)...")
    timm_model = load_tinyvit_model(
        model_name=args.model_name,
        models_dir=Path(args.models_dir),
        device=device,
    )

    sd = timm_model.state_dict()

    print("[DBG] Building SW model and loading SAME state_dict...")
    sw_model = TinyVitSWSmall().to(device)
    m2 = sw_model.load_state_dict(sd, strict=False)
    print("[DBG] SW load_state_dict(strict=False):")
    print(f"[DBG]   missing_keys   : {len(m2.missing_keys)}")
    print(f"[DBG]   unexpected_keys: {len(m2.unexpected_keys)}")

    timm_model.eval()
    sw_model.eval()

    # ------------------------
    # 3) Locate modules of interest
    # ------------------------
    target_bn_name = "stages.0.blocks.0.conv3.bn"
    target_block_name = "stages.0.blocks.0"
    target_conv3_name = "stages.0.blocks.0.conv3.conv"

    name_to_module_timm = dict(timm_model.named_modules())
    name_to_module_sw = dict(sw_model.named_modules())

    assert target_bn_name in name_to_module_timm, f"{target_bn_name} not in timm model"
    assert target_bn_name in name_to_module_sw, f"{target_bn_name} not in SW model"
    assert target_block_name in name_to_module_timm
    assert target_block_name in name_to_module_sw
    assert target_conv3_name in name_to_module_timm
    assert target_conv3_name in name_to_module_sw

    bn_ref = name_to_module_timm[target_bn_name]
    bn_sw = name_to_module_sw[target_bn_name]

    print("\n[DBG] BN basic info:")
    print(f"  bn_ref.training       : {bn_ref.training}")
    print(f"  bn_sw.training        : {bn_sw.training}")
    print(f"  bn_ref.track_running  : {bn_ref.track_running_stats}")
    print(f"  bn_sw.track_running   : {bn_sw.track_running_stats}")
    print(f"  bn_ref.eps            : {bn_ref.eps}")
    print(f"  bn_sw.eps             : {bn_sw.eps}")
    print(f"  bn_ref.momentum       : {bn_ref.momentum}")
    print(f"  bn_sw.momentum        : {bn_sw.momentum}")

    print("\n[DBG] BN parameter diffs (timm vs SW):")
    def param_diff(a, b, name):
        diff = (a - b).abs()
        print(f"  {name:14s}: max={diff.max().item():.6e} mean={diff.mean().item():.6e} shape={tuple(a.shape)}")

    param_diff(bn_ref.weight,       bn_sw.weight,       "weight")
    param_diff(bn_ref.bias,         bn_sw.bias,         "bias")
    param_diff(bn_ref.running_mean, bn_sw.running_mean, "running_mean")
    param_diff(bn_ref.running_var,  bn_sw.running_var,  "running_var")

    # ------------------------
    # 4) Hook conv3 + bn + block for both models
    # ------------------------
    activations = {
        "timm": {},
        "sw": {},
    }

    def make_hook(model_key, name):
        def hook_fn(module, inp, out):
            activations[model_key][name] = out.detach().cpu()
        return hook_fn

    hooks = []

    # timm hooks
    hooks.append(name_to_module_timm[target_conv3_name].register_forward_hook(
        make_hook("timm", "conv3")))
    hooks.append(bn_ref.register_forward_hook(
        make_hook("timm", "bn")))
    hooks.append(name_to_module_timm[target_block_name].register_forward_hook(
        make_hook("timm", "block")))

    # SW hooks
    hooks.append(name_to_module_sw[target_conv3_name].register_forward_hook(
        make_hook("sw", "conv3")))
    hooks.append(bn_sw.register_forward_hook(
        make_hook("sw", "bn")))
    hooks.append(name_to_module_sw[target_block_name].register_forward_hook(
        make_hook("sw", "block")))

    # ------------------------
    # 5) Forward both models on the SAME input
    # ------------------------
    with torch.no_grad():
        _ = timm_model(x)
        _ = sw_model(x)

    for h in hooks:
        h.remove()

    # Ensure we have everything
    for key in ["conv3", "bn", "block"]:
        assert key in activations["timm"], f"Missing timm {key} activation"
        assert key in activations["sw"], f"Missing sw {key} activation"

    conv3_ref = activations["timm"]["conv3"]
    conv3_sw  = activations["sw"]["conv3"]
    bn_ref_out = activations["timm"]["bn"]
    bn_sw_out  = activations["sw"]["bn"]
    blk_ref_out = activations["timm"]["block"]
    blk_sw_out  = activations["sw"]["block"]

    print("\n[DBG] Activation shape summary (timm vs SW):")
    print(f"  conv3_out : timm {tuple(conv3_ref.shape)}, sw {tuple(conv3_sw.shape)}")
    print(f"  bn_out    : timm {tuple(bn_ref_out.shape)}, sw {tuple(bn_sw_out.shape)}")
    print(f"  block_out : timm {tuple(blk_ref_out.shape)}, sw {tuple(blk_sw_out.shape)}")

    def act_diff(a, b, label):
        diff = (a - b).abs()
        print(f"  {label:10s}: max={diff.max().item():.6e} mean={diff.mean().item():.6e}")

    print("\n[DBG] Activation diffs (timm vs SW):")
    act_diff(conv3_ref, conv3_sw, "conv3_out")
    act_diff(bn_ref_out, bn_sw_out, "bn_out")
    act_diff(blk_ref_out, blk_sw_out, "block_out")

    # ------------------------
    # 6) Manual BN using bn_ref parameters & conv3_ref as input
    # ------------------------
    print("\n[DBG] Manual F.batch_norm using bn_ref params on conv3_ref...")

    x_bn = conv3_ref.to(device)
    manual_bn = F.batch_norm(
        x_bn,
        running_mean=bn_ref.running_mean.to(device),
        running_var=bn_ref.running_var.to(device),
        weight=bn_ref.weight.to(device),
        bias=bn_ref.bias.to(device),
        training=False,
        momentum=bn_ref.momentum,
        eps=bn_ref.eps,
    ).cpu()

    print("[DBG] Compare manual BN vs timm BN vs SW BN:")
    act_diff(manual_bn, bn_ref_out, "manual_vs_timm")
    act_diff(manual_bn, bn_sw_out,  "manual_vs_sw")

    # ------------------------
    # 7) Optional: compare against golden .bin for bn
    # ------------------------
    module_name_flat = target_bn_name.replace(".", "_")
    golden_fname = f"{module_name_flat}_output_{N}_golden.bin"
    golden_path = input_dir / golden_fname

    if golden_path.exists():
        print(f"\n[DBG] Found golden BN file: {golden_path}")
        with open(golden_path, "rb") as f:
            # golden format: int32 ndim, then dims, then float32 data
            ndim_arr = np.fromfile(f, dtype=np.int32, count=1)
            if ndim_arr.size != 1:
                raise RuntimeError("Failed to read ndim from golden file")
            ndim = int(ndim_arr[0])
            dims = np.fromfile(f, dtype=np.int32, count=ndim)
            data = np.fromfile(f, dtype=np.float32)
        expected = int(np.prod(dims))
        if data.size != expected:
            raise RuntimeError(
                f"Golden data size mismatch: got {data.size}, expected {expected}"
            )
        golden_bn = torch.from_numpy(data.reshape(*dims))

        print(f"[DBG] Golden BN shape: {tuple(golden_bn.shape)}")

        print("[DBG] Diffs vs golden:")
        act_diff(golden_bn, bn_ref_out, "golden_vs_timm")
        act_diff(golden_bn, bn_sw_out,  "golden_vs_sw")
        act_diff(golden_bn, manual_bn,  "golden_vs_manual")
    else:
        print(f"\n[DBG] No golden BN file found at: {golden_path}")
        print("      (If you want that check, re-run dump_tinyvit_activation.py "
              "for this module and N, then re-run this debug script.)")

    print("\n[DBG] Done. Use these stats to see exactly who agrees with whom "
          "(timm, SW, manual BN, golden).")


if __name__ == "__main__":
    main()
