import argparse
import os
import time
import random

import numpy as np
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

    return model


def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=4):
    """
    Returns train and test DataLoaders for CIFAR-10.
    Applies standard augmentation on the training set.
    """
    # Normalization values for CIFAR-10
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
    if len(history["train_loss"]) == 0:
        print("History is empty, skipping plots.")
        return

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
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileNetV2 on CIFAR-10: train or eval"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Run mode: train or eval (default: train)",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretraining",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="mobilenetv2_cifar10_best.pth",
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume/eval",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use AMP if GPU is available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def build_config_from_args(args):
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.label_smoothing = args.label_smoothing
    cfg.no_pretrained = args.no_pretrained
    cfg.save_path = args.save_path
    cfg.resume = args.resume
    cfg.mixed_precision = args.mixed_precision
    cfg.seed = args.seed
    cfg.mode = args.mode
    return cfg


def run_train(cfg, device):
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


def run_eval(cfg, device):
    print("Running in EVAL mode only.")

    # Only need test loader
    _, testloader = get_cifar10_loaders(
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

    # Determine which checkpoint to load
    ckpt_path = cfg.resume if cfg.resume else cfg.save_path
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No valid checkpoint found. "
            f"Provide --resume PATH or ensure {cfg.save_path} exists."
        )

    print(f"Loading checkpoint for eval from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # For eval, label smoothing typically 0 (but it's only for loss reporting)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    loss, acc1 = evaluate(model, criterion, testloader, device, epoch="EVAL")
    print(f"Final EVAL - Loss: {loss:.4f} | Top-1: {acc1:.2f}%")


def main():
    args = parse_args()
    cfg = build_config_from_args(args)

    # Reproducibility
    set_seed(cfg.seed)

    print("Config:", vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if cfg.mode == "train":
        run_train(cfg, device)
    else:  # "eval"
        run_eval(cfg, device)


if __name__ == "__main__":
    main()
