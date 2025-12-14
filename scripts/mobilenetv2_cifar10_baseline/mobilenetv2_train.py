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

# ---------------------------------
# Reproducibility Configuration
# ---------------------------------
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
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
    print(f"Random seed set to: {seed}")


def create_mobilenetv2_cifar10(num_classes=10, pretrained=True):
    """
    Create a MobileNetV2 model adapted for CIFAR-10 (32x32 RGB images).
    """
    if pretrained:
        print("Loading MobileNetV2 with ImageNet pretraining...")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    else:
        print("Loading MobileNetV2 without pretraining...")
        model = mobilenet_v2(weights=None)

    # Replace classifier to match CIFAR-10 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Optionally, adjust the first conv if needed; CIFAR-10 is 3x32x32 already,
    # so we keep the default (3-channel input) but note the smaller resolution.
    return model


def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=4):
    """
    Returns train and test DataLoaders for CIFAR-10.
    Applies standard augmentation on the training set.
    """
    # Normalization values for CIFAR-10 (approx or standard ImageNet-like)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader


def plot_curves(history, out_prefix="mobilenetv2_cifar10"):
    """
    Plot training and validation loss/accuracy curves and save them as PNGs.
    `history` is expected to be a dict with keys:
      - train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(len(history["train_loss"]))

    # Loss curves
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = f"{out_prefix}_loss.png"
    plt.savefig(loss_path)
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = f"{out_prefix}_accuracy.png"
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved loss curve to {loss_path}")
    print(f"Saved accuracy curve to {acc_path}")


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.
    Returns a list of accuracies corresponding to 'topk'.
    """
    if len(target.size()) > 1:
        target = target.squeeze()

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


def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    start_time = time.time()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

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
        f"Loss: {epoch_loss:.4f} | Top-1: {epoch_acc1:.2f}% "
        f"({elapsed:.1f}s)"
    )
    return epoch_loss, epoch_acc1


@torch.no_grad()
def evaluate(model, criterion, dataloader, device, epoch="TEST"):
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
    mixed_precision = False       # Use AMP if GPU is available


def main():
    # Reproducibility
    set_seed(42)

    cfg = Config()
    print("Config:", vars(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    trainloader, testloader = get_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Model
    model = create_mobilenetv2_cifar10(
        num_classes=10,
        pretrained=not cfg.no_pretrained,
    )
    model = model.to(device)

    # Optionally resume
    start_epoch = 0
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    if cfg.resume and os.path.isfile(cfg.resume):
        print(f"Loading checkpoint from {cfg.resume}")
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(
            f"Resumed from epoch {start_epoch} "
            f"with best acc {best_acc:.2f}%"
        )

    # Loss (with label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    # Cosine LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-4
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if (cfg.mixed_precision and device == "cuda")
        else None
    )

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n=== Epoch {epoch}/{cfg.epochs - 1} ===")
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, trainloader, device, epoch, scaler
        )
        val_loss, val_acc = evaluate(model, criterion, testloader, device, epoch)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                "epoch": epoch,
                "best_acc": best_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(state, cfg.save_path)
            print(
                f"New best acc: {best_acc:.2f}%. "
                f"Saved checkpoint to {cfg.save_path}."
            )

    print(f"\nTraining finished. Best Top-1 accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {cfg.save_path}")

    # Curves
    plot_curves(history, out_prefix="mobilenetv2_cifar10")
    print("Saved loss/accuracy curves to:")
    print(" mobilenetv2_cifar10_loss.png")
    print(" mobilenetv2_cifar10_accuracy.png")


if __name__ == "__main__":
    main()
