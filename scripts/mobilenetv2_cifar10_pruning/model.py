import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def create_mobilenetv2_cifar10(num_classes: int = 10, pretrained: bool = True):
    """
    MobileNetV2 backbone adapted to CIFAR-10:
    - First conv: 3x3, stride=1, padding=1 (no downsample).
    - Replace classifier head with dropout + Linear(num_classes).
    """
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = mobilenet_v2(weights=weights)
    else:
        backbone = mobilenet_v2(weights=None)

    backbone.features[0][0] = nn.Conv2d(
        3, 32, kernel_size=3, stride=1, padding=1, bias=False
    )

    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(backbone.last_channel, num_classes),
    )
    return backbone


def get_prunable_convs(model):
    """
    Return an ordered list of (name, module) for Conv2d layers we want to prune:
    1x1 convs with groups = 1.
    """
    prunable = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups == 1:
            prunable.append((name, m))
    return prunable


def channel_importance_l2(conv: nn.Conv2d):
    """
    Compute L2 norm of each output channel in a Conv2d layer.
    Returns a tensor of shape (out_channels,).
    """
    W = conv.weight.detach()   # [out_c, in_c, k_h, k_w]
    W_flat = W.view(W.size(0), -1)
    importance = torch.norm(W_flat, p=2, dim=1)
    return importance


def count_params_per_layer(convs):
    """
    convs: list of (name, conv_module)
    Returns dict {name: num_params_in_layer}
    """
    param_counts = {}
    for name, conv in convs:
        param_counts[name] = conv.weight.numel() + (
            conv.bias.numel() if conv.bias is not None else 0
        )
    return param_counts


def count_total_and_nonzero_params(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        if p is None:
            continue
        numel = p.numel()
        nz = (p != 0).sum().item()
        total += numel
        nonzero += nz
    sparsity = 1.0 - (nonzero / total)
    return total, nonzero, sparsity


def conv_summary(model, title: str = "Model"):
    """
    Print a small summary of prunable conv layers.
    """
    prunable = get_prunable_convs(model)
    param_counts = count_params_per_layer(prunable)
    print(f"=== {title} Conv2d Summary (1x1 convs, groups=1) ===")
    total_params = 0
    for name, conv in prunable:
        n_params = param_counts[name]
        total_params += n_params
        print(f"{name:40s}  out={conv.out_channels:4d}  params={n_params:7d}")
    print(f"Total conv params in these layers: {total_params}")
