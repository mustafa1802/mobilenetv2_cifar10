from __future__ import annotations
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import wandb


# ---------------------------------
# Reproducibility Configuration
# ---------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Sets seed for reproducibility across:
    - Python
    - NumPy
    - PyTorch (CPU + GPU)
    - cuDNN (deterministic)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure hash-based ops are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] Random seed set to: {seed}")


# ---------------------------------
# Model & Data
# ---------------------------------
def create_mobilenetv2_cifar10(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Create a MobileNetV2 model adapted for CIFAR-10.
    """
    if pretrained:
        print("[Model] Loading ImageNet weights")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    else:
        print("[Model] Training MobileNetV2 from scratch")
        model = mobilenet_v2(weights=None)

    # Replace classifier (last linear layer) for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_cifar10_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
):
    """
    Returns (trainloader, testloader) for CIFAR-10 with good augmentations.
    """
    # ImageNet-like normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # CIFAR-10 images are 32x32; we upscale to 224x224 for MobileNetV2
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return trainloader, testloader


# ---------------------------------
# Quantization Modules
# ---------------------------------
class MovingAveragePerChannelObserver(nn.Module):
    """
    Tracks a moving-average of per-channel max-abs values.
    Used to stabilize activation quantization (less noisy than per-batch).
    """
    def __init__(self, channel_dim: int, momentum: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.channel_dim = channel_dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_max", None)  # shape: 1 x C x 1 x 1 (or similar)
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # Compute current per-channel max-abs over all non-channel dims
        dims = [d for d in range(x.dim()) if d != self.channel_dim]
        cur_max = x.detach().abs().amax(dim=dims, keepdim=True)

        if not self.initialized:
            self.running_max = cur_max
            self.initialized.fill_(True)
        else:
            self.running_max = (
                (1.0 - self.momentum) * self.running_max
                + self.momentum * cur_max
            )

        # avoid zeros for scale computation
        self.running_max = torch.clamp(self.running_max, min=self.eps)

    @torch.no_grad()
    def get_scale(self, bit_width: int) -> torch.Tensor:
        qmax = 2 ** (bit_width - 1) - 1
        return self.running_max / qmax


class PerChannelWeightFakeQuant(nn.Module):
    """
    Per-channel symmetric fake quantization for weights.
    Uses the current weight tensor statistics (no moving-average observer).
    """
    def __init__(self, bit_width: int = 8, channel_dim: int = 0, eps: float = 1e-8):
        super().__init__()
        assert bit_width >= 2, "bit_width must be >= 2"
        self.bit_width = bit_width
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # per-channel max-abs over all non-channel dims
            dims = [d for d in range(w.dim()) if d != self.channel_dim]
            max_val = w.detach().abs().amax(dim=dims, keepdim=True)
            max_val = torch.clamp(max_val, min=self.eps)

            qmax = 2 ** (self.bit_width - 1) - 1
            scale = max_val / qmax

            w_int = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
            w_dequant = w_int * scale

        # STE: forward uses quantized value, backward ~ identity
        w_ste = w + (w_dequant - w).detach()
        return w_ste


class PerChannelActivationFakeQuant(nn.Module):
    """
    Per-channel symmetric fake quantization for activations
    using a moving-average per-channel observer + STE.
    """
    def __init__(self, bit_width: int = 8, channel_dim: int = 1, momentum: float = 0.1):
        super().__init__()
        assert bit_width >= 2, "bit_width must be >= 2"
        self.bit_width = bit_width
        self.channel_dim = channel_dim
        self.observer = MovingAveragePerChannelObserver(
            channel_dim=channel_dim,
            momentum=momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # update observer from current batch
        self.observer.update(x)

        with torch.no_grad():
            scale = self.observer.get_scale(self.bit_width)
            qmax = 2 ** (self.bit_width - 1) - 1
            x_int = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
            x_dequant = x_int * scale

        # STE: forward uses quantized value, backward ~ identity
        x_ste = x + (x_dequant - x).detach()
        return x_ste


class QuantizedConv2d(nn.Module):
    """
    Wraps nn.Conv2d to apply per-channel fake weight + activation quantization.
    - Weights: per-output-channel (dim=0)
    - Activations: per-input-channel (NCHW, dim=1)
    """
    def __init__(self, conv: nn.Conv2d, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        # copy original conv hyperparameters
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.bias_flag = conv.bias is not None

        # actual conv that holds params
        self.conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=self.bias_flag,
        )

        # copy weights/bias
        self.conv.weight.data.copy_(conv.weight.data)
        if self.bias_flag:
            self.conv.bias.data.copy_(conv.bias.data)

        # weights: per-output-channel (dim=0)
        self.w_fake_quant = PerChannelWeightFakeQuant(
            bit_width=w_bits,
            channel_dim=0,
        )
        # activations: per-channel (NCHW -> dim=1) with moving-average observer
        self.a_fake_quant = PerChannelActivationFakeQuant(
            bit_width=a_bits,
            channel_dim=1,
            momentum=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # activation per-channel fake quant
        x = self.a_fake_quant(x)

        # weight per-channel fake quant
        w_q = self.w_fake_quant(self.conv.weight)
        b = self.conv.bias
        x = nn.functional.conv2d(
            x, w_q, b,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return x


class QuantizedLinear(nn.Module):
    """
    Wraps nn.Linear to apply per-channel fake weight + activation quantization.
    - Weights: per-output-channel (dim=0)
    - Activations: per-feature (dim=1) with moving-average observer.
    """
    def __init__(self, linear: nn.Linear, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        self.linear = nn.Linear(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
        )
        self.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            self.linear.bias.data.copy_(linear.bias.data)

        # weights: [out_features, in_features] -> dim=0
        self.w_fake_quant = PerChannelWeightFakeQuant(
            bit_width=w_bits,
            channel_dim=0,
        )
        # activations: [N, F] -> dim=1 with moving-average observer
        self.a_fake_quant = PerChannelActivationFakeQuant(
            bit_width=a_bits,
            channel_dim=1,
            momentum=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a_fake_quant(x)
        w_q = self.w_fake_quant(self.linear.weight)
        b = self.linear.bias
        return nn.functional.linear(x, w_q, b)


def apply_qat_wrapping(
    model: nn.Module,
    w_bits: int = 8,
    a_bits: int = 8,
    skip_first_last: bool = False,
) -> nn.Module:
    """
    Recursively replace Conv2d / Linear with QuantizedConv2d / QuantizedLinear.

    If skip_first_last=True:
      - do NOT quantize the very first Conv2d encountered
      - do NOT quantize the very last Linear encountered
    """

    # Count total convs/linears so we know which ones are "first" and "last"
    total_convs = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    total_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    counters = {"conv": 0, "linear": 0}

    def _wrap(module: nn.Module):
        for name, child in list(module.named_children()):
            # Recurse first
            _wrap(child)

            # Then possibly replace leaf modules
            if isinstance(child, nn.Conv2d):
                counters["conv"] += 1

                if skip_first_last:
                    # Skip the very first conv
                    if counters["conv"] == 1:
                        continue
                # Quantize all other convs
                setattr(module, name, QuantizedConv2d(child, w_bits, a_bits))

            elif isinstance(child, nn.Linear):
                counters["linear"] += 1

                if skip_first_last:
                    # Skip the very last linear
                    if counters["linear"] == total_linears:
                        continue
                # Quantize all other linears
                setattr(module, name, QuantizedLinear(child, w_bits, a_bits))

    _wrap(model)
    return model


def estimate_model_size_mb(
    model: nn.Module,
    weight_bit_width: int = 32,
    quantize_first_last: bool = True,
) -> float:
    """
    Estimate model *weight* storage size (MB), accounting for quantization modes.

    - All params are assumed to be 32-bit by default.
    - Conv2d / Linear *weights* can be quantized to `weight_bit_width`.
    - Biases and all non-conv/linear params (e.g., BatchNorm) stay 32-bit.
    - If `quantize_first_last` is False, we treat:
        - first Conv2d as FP32
        - last Linear as FP32
    """

    # --- 0. Baseline: everything fp32 ---
    total_params = sum(p.numel() for p in model.parameters())
    base_bits = total_params * 32  # all params assumed 32-bit

    # If no quantization or same bit-width, just return baseline
    if weight_bit_width >= 32:
        total_bytes = base_bits / 8.0
        return total_bytes / (1024 ** 2)

    # --- 1. Count how many Conv/Linear *weights* will be quantized ---
    total_convs = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    total_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    conv_counter = 0
    linear_counter = 0
    quantized_weight_params = 0  # number of *weights* (not biases) to be quantized

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_counter += 1

            # Decide if THIS conv is quantized
            is_quant = True
            if not quantize_first_last:
                # Skip *very first* conv
                if conv_counter == 1:
                    is_quant = False

            if is_quant:
                quantized_weight_params += module.weight.numel()

        elif isinstance(module, nn.Linear):
            linear_counter += 1

            # Decide if THIS linear is quantized
            is_quant = True
            if not quantize_first_last:
                # Skip *very last* linear
                if linear_counter == total_linears:
                    is_quant = False

            if is_quant:
                quantized_weight_params += module.weight.numel()

    # --- 2. Adjust baseline size for those quantized weights ---
    bit_savings_per_param = 32 - weight_bit_width
    savings_bits = quantized_weight_params * bit_savings_per_param

    quantized_bits = base_bits - savings_bits
    total_bytes = quantized_bits / 8.0
    return total_bytes / (1024 ** 2)


# ---------------------------------
# Training Utils
# ---------------------------------
def plot_curves(history, out_prefix: str = "mobilenetv2_cifar10") -> None:
    """
    Plot train/val loss and accuracy curves and save them as PNGs.
    history: dict with keys:
        'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    epochs = range(len(history["train_loss"]))

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val/Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.savefig(f"{out_prefix}_loss.png", bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Top-1")
    plt.plot(epochs, history["val_acc"], label="Val/Test Top-1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.savefig(f"{out_prefix}_accuracy.png", bbox_inches="tight")
    plt.close()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = preds.size(1)
        log_preds = torch.log_softmax(preds, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the top-k accuracy for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k indices
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # Compare with targets expanded
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader,
    device: str,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    start_time = time.time()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        top1, = accuracy(outputs, targets, topk=(1,))
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        total += bs

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(dataloader)}] "
                f"Loss: {running_loss / total:.4f} | "
                f"Top-1: {running_top1 / total:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc1 = running_top1 / total
    elapsed = time.time() - start_time
    print(
        f"Epoch [{epoch}] TRAIN - "
        f"Loss: {epoch_loss:.4f} | Top-1: {epoch_acc1:.2f}% | "
        f"Time: {elapsed:.1f}s"
    )
    return epoch_loss, epoch_acc1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    device: str,
    epoch: int | str = "TEST",
):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        top1, = accuracy(outputs, targets, topk=(1,))
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        total += bs

    loss = running_loss / total
    acc1 = running_top1 / total
    print(f"Epoch [{epoch}] VALID - Loss: {loss:.4f} | Top-1: {acc1:.2f}%")
    return loss, acc1


# ---------------------------------
# Config + Training Driver
# ---------------------------------
class Config:
    def __init__(self):
        # Data / training
        self.data_dir = "./data"
        self.epochs = 5
        self.batch_size = 128
        self.lr = 0.05
        self.weight_decay = 4e-5
        self.num_workers = 4
        self.label_smoothing = 0.1
        self.no_pretrained = False
        self.save_path = "mobilenetv2_cifar10_best_compressed.pth"
        self.resume = ""
        self.mixed_precision = False

        # QAT / compression-related options
        self.use_qat = True
        self.weight_bit_width = 8
        self.activation_bit_width = 8
        self.quantize_first_last = True

        # Optional fp32 checkpoint path (for starting from pruned/finetuned model)
        self.fp32_checkpoint = ""

        # W&B options
        self.wandb_project = "mobilenetv2_cifar10_qat"
        self.wandb_entity = None
        self.use_wandb = True


def train_and_eval(run_cfg: Config):
    # -------------------------
    # Optional W&B run init
    # -------------------------
    if run_cfg.use_wandb:
        wandb.init(
            project=run_cfg.wandb_project,
            entity=run_cfg.wandb_entity,
            config={
                "epochs": run_cfg.epochs,
                "batch_size": run_cfg.batch_size,
                "lr": run_cfg.lr,
                "weight_decay": run_cfg.weight_decay,
                "weight_bit_width": run_cfg.weight_bit_width,
                "activation_bit_width": run_cfg.activation_bit_width,
                "use_qat": run_cfg.use_qat,
                "quantize_first_last": run_cfg.quantize_first_last,
            },
        )
        wb_cfg = wandb.config
        # Keep cfg and wandb.config aligned (sweep may override)
        run_cfg.lr = wb_cfg.lr
        run_cfg.weight_bit_width = wb_cfg.weight_bit_width
        run_cfg.activation_bit_width = wb_cfg.activation_bit_width
        run_cfg.weight_decay = wb_cfg.weight_decay

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using device: {device}")

    # Data
    trainloader, testloader = get_cifar10_loaders(
        data_dir=run_cfg.data_dir,
        batch_size=run_cfg.batch_size,
        num_workers=run_cfg.num_workers,
    )

    # Model
    model = create_mobilenetv2_cifar10(
        num_classes=10,
        pretrained=not run_cfg.no_pretrained,
    )

    # Optional pretrained fp32 model weights before applying QAT
    if run_cfg.fp32_checkpoint:
        print(f"[Model] Loading pretrained checkpoint from: {run_cfg.fp32_checkpoint}")
        state_dict = torch.load(run_cfg.fp32_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    # Baseline FP32 model size (for compression ratio)
    model_size_mb_fp32 = estimate_model_size_mb(
        model,
        weight_bit_width=32,
        quantize_first_last=True,
    )

    # Apply QAT wrapping if requested
    if run_cfg.use_qat:
        print(
            f"[QAT] Applying QAT: weight_bit_width={run_cfg.weight_bit_width}, "
            f"activation_bit_width={run_cfg.activation_bit_width}, "
            f"quantize_first_last={run_cfg.quantize_first_last}"
        )
        model = apply_qat_wrapping(
            model,
            w_bits=run_cfg.weight_bit_width,
            a_bits=run_cfg.activation_bit_width,
            skip_first_last=not run_cfg.quantize_first_last,
        )

    model = model.to(device)

    # Criterion / optimizer / scheduler
    criterion = LabelSmoothingCrossEntropy(run_cfg.label_smoothing)

    # Weight decay on weights, none on bias/norm
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

    optimizer = optim.SGD(
        [
            {"params": decay, "weight_decay": run_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=run_cfg.lr,
        momentum=0.9,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=run_cfg.epochs, eta_min=1e-4
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if (run_cfg.mixed_precision and device == "cuda")
        else None
    )

    start_epoch = 0
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Estimated quantized model size and compression ratio (only depends on bit-width)
    if run_cfg.use_qat:
        model_size_mb_quant = estimate_model_size_mb(
            model,
            weight_bit_width=run_cfg.weight_bit_width,
            quantize_first_last=run_cfg.quantize_first_last,
        )
        compression_ratio = model_size_mb_fp32 / max(model_size_mb_quant, 1e-8)
    else:
        model_size_mb_quant = model_size_mb_fp32
        compression_ratio = 1.0

    print(f"[Size] Estimated FP32 model size: {model_size_mb_fp32:.3f} MB")
    print(f"[Size] Estimated quantized model size: {model_size_mb_quant:.3f} MB")
    print(f"[Size] Compression ratio (FP32 / quantized): {compression_ratio:.2f}x")

    for epoch in range(start_epoch, run_cfg.epochs):
        print(f"\n=== Epoch {epoch}/{run_cfg.epochs - 1} ===")
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, trainloader, device, epoch, scaler
        )
        val_loss, val_acc = evaluate(model, criterion, testloader, device, epoch)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # -------------------------
        # W&B logging per epoch
        # -------------------------
        if run_cfg.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "weight_bit_width": run_cfg.weight_bit_width,
                    "activation_bit_width": run_cfg.activation_bit_width,
                    "model_size_mb": model_size_mb_quant,
                    "compression_ratio": compression_ratio,
                    # for W&B parallel coordinates plot
                    "test_accuracy": val_acc,
                }
            )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(state, run_cfg.save_path)
            print(
                f"New best acc: {best_acc:.2f}%. "
                f"Saved checkpoint to {run_cfg.save_path}."
            )

    print(f"\n[Result] Training finished. Best Top-1 accuracy: {best_acc:.2f}%")
    print(f"[Result] Best model saved to: {run_cfg.save_path}")

    # final history plots (optional, only for the last run)
    plot_curves(history, out_prefix="mobilenetv2_cifar10")

    if run_cfg.use_wandb:
        wandb.summary["best_val_acc"] = best_acc
        wandb.summary["model_size_mb"] = model_size_mb_quant
        wandb.summary["compression_ratio"] = compression_ratio
        wandb.finish()

    return history, best_acc


# ---------------------------------
# W&B Sweep configuration & helper
# ---------------------------------
SWEEP_CONFIG = {
    "method": "random",
    "metric": {
        "name": "test_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "weight_bit_width": {
            "values": [2, 4, 8],
        },
        "activation_bit_width": {
            "values": [2, 4, 8],
        },
        "lr": {
            "values": [0.001],
        },
        "weight_decay": {
            "values": [4e-5],
        },
        "quantize_first_last": {
            "values": [True, False],
        },
    },
}


def sweep_train():
    """
    One sweep run: W&B will populate wandb.config, then we create a Config
    and run train_and_eval with those hyperparams.
    """
    run = wandb.init()
    wb_cfg = wandb.config

    cfg = Config()
    cfg.weight_bit_width = wb_cfg.weight_bit_width
    cfg.activation_bit_width = wb_cfg.activation_bit_width
    cfg.lr = wb_cfg.lr
    cfg.weight_decay = wb_cfg.weight_decay
    cfg.quantize_first_last = wb_cfg.quantize_first_last

    print("\n[SWEEP RUN CONFIG]")
    print(f"weight_bit_width: {cfg.weight_bit_width}")
    print(f"activation_bit_width: {cfg.activation_bit_width}")
    print(f"lr: {cfg.lr}")
    print(f"weight_decay: {cfg.weight_decay}")
    print(f"quantize_first_last: {cfg.quantize_first_last}")

    train_and_eval(cfg)


# ---------------------------------
# CLI / main
# ---------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileNetV2 CIFAR-10 QAT training script"
    )
    # Data / training
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--save-path", type=str, default="mobilenetv2_cifar10_qat_4w8a.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    # QAT
    parser.add_argument("--use-qat", action="store_true", default=True)
    parser.add_argument("--weight-bit-width", type=int, default=4)
    parser.add_argument("--activation-bit-width", type=int, default=8)
    parser.add_argument(
        "--no-quantize-first-last",
        action="store_true",
        help="If set, first conv & last linear remain FP32.",
    )

    # Optional fp32 checkpoint
    parser.add_argument(
        "--fp32-checkpoint",
        type=str,
        default="",
        help="Path to fp32 (pruned/finetuned) checkpoint to start from.",
    )

    # W&B
    parser.add_argument("--wandb-project", type=str, default="mobilenetv2_cifar10_qat")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B logging (requires WANDB_API_KEY to be set).",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "sweep"],
        default="train",
        help="Run a single training job or a W&B sweep agent.",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=16,
        help="Number of sweep runs when mode='sweep'.",
    )

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "sweep":
        # W&B sweep mode
        if not args.use_wandb:
            raise ValueError("Sweep mode requires --use-wandb.")
        print("[Main] Creating W&B sweep...")
        sweep_id = wandb.sweep(
            sweep=SWEEP_CONFIG,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print("Created W&B sweep with id:", sweep_id)
        wandb.agent(
            sweep_id,
            function=sweep_train,
            count=args.sweep_count,
        )
    else:
        # Single QAT fine-tuning run
        cfg = Config()

        # Fill in config from args (overriding defaults)
        cfg.data_dir = args.data_dir
        cfg.epochs = args.epochs
        cfg.batch_size = args.batch_size
        cfg.lr = args.lr
        cfg.weight_decay = args.weight_decay
        cfg.label_smoothing = args.label_smoothing
        cfg.no_pretrained = args.no_pretrained
        cfg.save_path = args.save_path
        cfg.resume = args.resume
        cfg.mixed_precision = args.mixed_precision
        cfg.num_workers = args.num_workers

        cfg.use_qat = args.use_qat
        cfg.weight_bit_width = args.weight_bit_width
        cfg.activation_bit_width = args.activation_bit_width
        cfg.quantize_first_last = not args.no_quantize_first_last

        cfg.fp32_checkpoint = args.fp32_checkpoint

        cfg.wandb_project = args.wandb_project
        cfg.wandb_entity = args.wandb_entity
        cfg.use_wandb = args.use_wandb

        print("[Config]", vars(cfg))

        history, best_acc = train_and_eval(cfg)
        print("Best QAT accuracy:", best_acc)


if __name__ == "__main__":
    main()
