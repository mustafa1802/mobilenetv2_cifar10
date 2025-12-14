#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights



def create_mobilenetv2_cifar10(num_classes=10, pretrained=True):
    """
    Create a MobileNetV2 model adapted for CIFAR-10.
    """
    if pretrained:
        print("Load ImageNet weights")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)

    # Replace classifier (last linear layer) for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model



class Config:
    data_dir = "./data"           # For Colab, use "/content/data"
    epochs = 30
    batch_size = 128
    lr = 0.05
    weight_decay = 4e-5
    num_workers = 4
    label_smoothing = 0.1
    no_pretrained = False         # Set True to disable ImageNet pretraining
    save_path = "mobilenetv2_cifar10_best.pth"
    resume = ""                   # Path to checkpoint, or "" to start fresh
    mixed_precision = False        # Use AMP if GPU is available

cfg = Config()
print("Config:", vars(cfg))



PRUNE_FRACTIONS = {
    "features.1.conv.1":   0.10,
    "features.2.conv.0.0": 0.10,
    "features.2.conv.2":   0.10,
    "features.3.conv.0.0": 0.10,
    "features.3.conv.2":   0.10,
    "features.4.conv.0.0": 0.20,
    "features.4.conv.2":   0.20,
    "features.5.conv.0.0": 0.20,
    "features.5.conv.2":   0.20,
    "features.6.conv.0.0": 0.20,
    "features.6.conv.2":   0.20,
    "features.7.conv.0.0": 0.20,
    "features.7.conv.2":   0.20,
    "features.8.conv.0.0": 0.30,
    "features.8.conv.2":   0.30,
    "features.9.conv.0.0": 0.30,
    "features.9.conv.2":   0.30,
    "features.10.conv.0.0": 0.30,
    "features.10.conv.2":   0.30,
    "features.11.conv.0.0": 0.30,
    "features.11.conv.2":   0.28,
    "features.12.conv.0.0": 0.30,
    "features.12.conv.2":   0.30,
    "features.13.conv.0.0": 0.30,
    "features.13.conv.2":   0.30,
    "features.14.conv.0.0": 0.35,
    "features.14.conv.2":   0.27,
    "features.15.conv.0.0": 0.32,
    "features.15.conv.2":   0.35,
    "features.16.conv.0.0": 0.35,
    "features.16.conv.2":   0.35,
    "features.17.conv.0.0": 0.35,
    "features.17.conv.2":   0.35,
    "features.18.0":        0.20,
}



def collect_activation_sizes_baseline(model: nn.Module,
                                      input_tensor: torch.Tensor):
    """
    Runs a forward pass on the *baseline* (unpruned) model and
    returns a dict: {module_name: num_elements_in_output_activation}.
    """
    act_sizes = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                act_sizes[name] = out.numel()
            elif isinstance(out, (tuple, list)):
                act_sizes[name] = sum(
                    o.numel() for o in out if isinstance(o, torch.Tensor)
                )
        return hook

    # Attach hooks to conv/linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Clean up
    for h in handles:
        h.remove()

    return act_sizes



def compute_runtime_activation_compression_with_table(
    baseline_model: nn.Module,
    dummy_input: torch.Tensor,
    prune_fractions: dict,
    activation_bit_width: int = 8,
):
    """
    Computes:
      - overall runtime activation compression vs baseline FP32
      - per-layer table with prune fraction and per-layer compression

    Returns:
      compression_ratio (float),
      layer_rows (list of dicts)
    """
    # 1) Collect baseline activation sizes
    act_sizes = collect_activation_sizes_baseline(baseline_model, dummy_input)

    bits_baseline_total = 0.0
    bits_pruned_quant_total = 0.0
    layer_rows = []

    for name, N in act_sizes.items():
        # Baseline bits (FP32)
        bits_baseline = N * 32

        # Prune fraction (0.0 if not pruned / not in dict)
        p_l = prune_fractions.get(name, 0.0)

        # Pruned + quantized bits
        bits_pruned_quant = (1.0 - p_l) * N * activation_bit_width

        bits_baseline_total += bits_baseline
        bits_pruned_quant_total += bits_pruned_quant

        # Per-layer compression: baseline FP32 -> pruned + quantized
        if bits_pruned_quant > 0:
            layer_compression = bits_baseline / bits_pruned_quant
        else:
            layer_compression = float("inf")

        layer_rows.append({
            "layer": name,
            "N_elements": N,
            "prune_fraction": p_l,
            "baseline_bits": bits_baseline,
            "pruned_quant_bits": bits_pruned_quant,
            "compression": layer_compression,
        })

    overall_compression = bits_baseline_total / max(bits_pruned_quant_total, 1e-8)
    return overall_compression, layer_rows



# 1) Build your *baseline* (unpruned) MobileNetV2 for CIFAR-10
# Model
baseline_model = create_mobilenetv2_cifar10(
    num_classes=10,
    pretrained=not cfg.no_pretrained,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
baseline_model = baseline_model.to(device)

# 2) Representative input
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# 3) Compute overall compression + per-layer stats
overall_act_compression, layer_rows = compute_runtime_activation_compression_with_table(
    baseline_model=baseline_model,
    dummy_input=dummy_input,
    prune_fractions=PRUNE_FRACTIONS,
    activation_bit_width=8,
)

print(f"\nOverall runtime activation compression vs baseline FP32: "
      f"{overall_act_compression:.2f}x\n")

# 4) Print a small per-layer table
print(f"{'Layer':40s} {'Prune':>7s} {'Comp(x)':>8s}")
print("-" * 60)
for row in layer_rows:
    name = row["layer"]
    # Only print layers that are in your prune dict (or print all if you prefer)
    if name in PRUNE_FRACTIONS:
        p_l = row["prune_fraction"]
        comp = row["compression"]
        print(f"{name:40s} {p_l:7.2f} {comp:8.2f}")

