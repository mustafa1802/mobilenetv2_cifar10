import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import create_mobilenetv2_cifar10
from data import get_cifar10_loaders


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate_with_criterion(model, dataloader, criterion, device):
    """
    Evaluate the model on dataloader using the given criterion, returning (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss = total_loss / total
    train_acc = 100.0 * correct / total
    elapsed = time.time() - start
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% ({elapsed:.1f}s)")
    return train_loss, train_acc


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def train_initial_model(cfg, device):
    """
    Train baseline MobileNetV2 on CIFAR-10 or return existing checkpoint.
    """
    trainloader, testloader = get_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = create_mobilenetv2_cifar10(num_classes=10, pretrained=cfg.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.base_lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            running_correct += predicted.eq(labels).sum().item()

        lr_scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * running_correct / total
        test_loss, test_acc = evaluate(model, testloader, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% "
            f"({elapsed:.1f}s)"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(
            {"state_dict": best_state, "best_acc": best_acc},
            cfg.base_ckpt_path,
        )
        print(f"Saved best model with accuracy: {best_acc:.2f}%")
    else:
        torch.save(
            {"state_dict": model.state_dict(), "best_acc": best_acc},
            cfg.base_ckpt_path,
        )
        print("Saved final model without improvement record.")

    return model, best_acc


def estimate_layer_sensitivity(model, criterion, device, calib_loader, base_prune=0.1):
    """
    For each prunable layer:
    - Create a pruned copy of the model with small prune fraction (base_prune).
    - Measure the loss difference vs. original model on calib_loader.
    """
    from model import get_prunable_convs  # local import to avoid cycles

    model.eval()
    prunable = get_prunable_convs(model)

    base_loss, _ = evaluate_with_criterion(model, calib_loader, criterion, device)
    print(f"Baseline calibration loss: {base_loss:.4f}")

    sensitivities = {}

    for name, _conv in prunable:
        print(f"\n[Sensitivity] Testing layer: {name}")

        model_copy = create_mobilenetv2_cifar10(num_classes=10, pretrained=False)
        model_copy.load_state_dict(model.state_dict())

        layer_dict = dict(model_copy.named_modules())
        target_conv = layer_dict[name]

        from model import channel_importance_l2
        importance = channel_importance_l2(target_conv)
        num_channels = importance.shape[0]
        k = max(1, int(base_prune * num_channels))
        sorted_idx = torch.argsort(importance)
        prune_idx = sorted_idx[:k]

        mask = torch.ones(num_channels, dtype=torch.bool)
        mask[prune_idx] = False

        with torch.no_grad():
            W = target_conv.weight.data
            W_shape = W.shape
            W_flat = W.view(W.size(0), -1)
            W_flat[mask == 0, :] = 0.0
            target_conv.weight.data = W_flat.view(W_shape)

            if target_conv.bias is not None:
                b = target_conv.bias.data
                b[mask == 0] = 0.0
                target_conv.bias.data = b

        model_copy = model_copy.to(device)
        sens_loss, sens_acc = evaluate_with_criterion(
            model_copy, calib_loader, criterion, device
        )

        sens_delta = sens_loss - base_loss
        sensitivities[name] = sens_delta
        print(
            f"  => Sensitivity (loss difference): {sens_delta:.4f} "
            f"(prune={base_prune*100:.1f}%) | "
            f"Calib Acc: {sens_acc:.2f}%"
        )

    return sensitivities
