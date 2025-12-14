import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # Data / loader
    data_dir = "./data"
    batch_size = 128
    num_workers = 2

    # Baseline training
    epochs = 30
    base_lr = 0.05
    weight_decay = 5e-4
    pretrained = True

    # Sensitivity estimation
    label_smoothing = 0.1
    calib_samples = 1024
    calib_batch_size = 128

    # Global pruning targets
    target_prunes = [0.10, 0.20, 0.30, 0.40, 0.50]

    # Fine-tuning after each pruning stage
    finetune_epochs = 5
    finetune_lr = 0.01
    finetune_weight_decay = 5e-4

    # Final fine-tuning of best pruned model
    final_finetune_epochs = 15
    final_finetune_lr = 0.005
    final_finetune_weight_decay = 5e-4

    # Checkpoint names
    base_ckpt_path = "mobilenetv2_cifar10_base.pth"
    best_overall_ckpt = "mobilenetv2_cifar10_pruned_best_overall.pth"
    final_pruned_ckpt = "mobilenetv2_cifar10_pruned_finetuned.pth"
