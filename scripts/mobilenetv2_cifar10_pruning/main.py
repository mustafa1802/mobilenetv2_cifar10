import os
import torch

from config import Config, set_seed
from data import get_cifar10_loaders, make_calib_loader
from model import (
    create_mobilenetv2_cifar10,
    get_prunable_convs,
    conv_summary,
    count_params_per_layer,
)
from pruning import (
    compute_prunability_scores,
    allocate_prune_fractions,
    apply_structured_channel_pruning,
)
from train import (
    train_initial_model,
    LabelSmoothingCrossEntropy,
    estimate_layer_sensitivity,
    evaluate_with_criterion,
    train_one_epoch,
)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = Config()

    # -----------------------------
    # Stage 1: Train / load baseline
    # -----------------------------
    print("=== Stage 1: Train or load baseline MobileNetV2 on CIFAR-10 ===")
    if os.path.exists(cfg.base_ckpt_path):
        print(f"Found existing checkpoint at {cfg.base_ckpt_path}, loading...")
        checkpoint = torch.load(cfg.base_ckpt_path, map_location="cpu")
        model = create_mobilenetv2_cifar10(num_classes=10, pretrained=cfg.pretrained)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        base_acc = checkpoint.get("best_acc", None)
        if base_acc is not None:
            print(f"Loaded baseline model with accuracy: {base_acc:.2f}%")
    else:
        print("No baseline checkpoint found, training from scratch...")
        model, base_acc = train_initial_model(cfg, device)

    trainloader, testloader = get_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # -----------------------------
    # Stage 2: Sensitivity analysis
    # -----------------------------
    print("\n=== Stage 2: Estimate layer sensitivities with label smoothing loss ===")
    calib_loader = make_calib_loader(
        test_dataset=testloader.dataset,
        batch_size=cfg.calib_batch_size,
        num_samples=cfg.calib_samples,
    )
    criterion_ls = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    sensitivities = estimate_layer_sensitivity(
        model=model,
        criterion=criterion_ls,
        device=device,
        calib_loader=calib_loader,
        base_prune=0.10,
    )

    # -----------------------------
    # Stage 3: Prunability scores
    # -----------------------------
    print("\n=== Stage 3: Compute prunability scores based on sensitivities ===")
    scores = compute_prunability_scores(sensitivities)

    # -----------------------------
    # Stage 4: Multi-stage pruning + fine-tuning
    # -----------------------------
    print("\n=== Stage 4: Multi-stage global channel pruning + fine-tuning ===")
    prunable_convs = get_prunable_convs(model)
    param_counts = count_params_per_layer(prunable_convs)

    conv_summary(model, title="Baseline")

    best_overall_acc = 0.0
    best_overall_state = None
    best_stage = None

    for target_global in cfg.target_prunes:
        print(f"\n--- Target global prune fraction: {target_global:.2f} ---")

        # Make a fresh copy of baseline model each stage
        m_stage = create_mobilenetv2_cifar10(num_classes=10, pretrained=False)
        m_stage.load_state_dict(model.state_dict())

        prune_fracs = allocate_prune_fractions(
            scores=scores,
            param_counts=param_counts,
            target_global_prune=target_global,
        )

        m_stage, masks = apply_structured_channel_pruning(m_stage, prunable_convs, prune_fracs)
        m_stage = m_stage.to(device)

        train_loss, train_acc = evaluate_with_criterion(
            m_stage, trainloader, criterion_ls, device
        )
        val_loss, val_acc = evaluate_with_criterion(
            m_stage, testloader, criterion_ls, device
        )
        print(f"  After pruning only: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        optimizer = torch.optim.SGD(
            m_stage.parameters(),
            lr=cfg.finetune_lr,
            momentum=0.9,
            weight_decay=cfg.finetune_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.finetune_epochs
        )

        best_ft_acc = 0.0
        best_ft_state = None

        for epoch in range(1, cfg.finetune_epochs + 1):
            train_one_epoch(
                m_stage,
                criterion_ls,
                optimizer,
                trainloader,
                device,
                epoch,
                scheduler=scheduler,
            )
            _, stage_val_acc = evaluate_with_criterion(
                m_stage, testloader, criterion_ls, device
            )
            print(f"    [Fine-tune] Epoch {epoch}: Val Acc={stage_val_acc:.2f}%")

            if stage_val_acc > best_ft_acc:
                best_ft_acc = stage_val_acc
                best_ft_state = {k: v.cpu() for k, v in m_stage.state_dict().items()}

        if best_ft_state is not None:
            ckpt_name = f"mobilenetv2_cifar10_pruned_{int(target_global*100)}.pth"
            torch.save(
                {"state_dict": best_ft_state, "best_acc": best_ft_acc},
                ckpt_name,
            )
            print(
                f"  => Saved best pruned model at stage {target_global:.2f} "
                f"with Val Acc={best_ft_acc:.2f}% (file={ckpt_name})"
            )

        if best_ft_acc > best_overall_acc:
            best_overall_acc = best_ft_acc
            best_overall_state = best_ft_state
            best_stage = target_global

    if best_overall_state is not None:
        torch.save(
            {
                "state_dict": best_overall_state,
                "best_acc": best_overall_acc,
                "best_stage": best_stage,
            },
            cfg.best_overall_ckpt,
        )
        print(
            f"\n[Summary] Best overall pruned model: "
            f"stage={best_stage:.2f}, val_acc={best_overall_acc:.2f}%"
        )
    else:
        print("\n[Summary] No improved pruned model found.")

    # -----------------------------
    # Stage 5: Final fine-tuning of best pruned model
    # -----------------------------
    print("\n=== Stage 5: Final fine-tuning of best pruned model ===")
    if best_overall_state is not None:
        model.load_state_dict(best_overall_state)
        model = model.to(device)
    else:
        print("Warning: Using baseline model for final fine-tuning (no pruning improvements).")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.final_finetune_lr,
        momentum=0.9,
        weight_decay=cfg.final_finetune_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.final_finetune_epochs
    )

    best_ft_acc = 0.0
    best_ft_state = None

    for epoch in range(1, cfg.final_finetune_epochs + 1):
        train_one_epoch(
            model,
            criterion_ls,
            optimizer,
            trainloader,
            device,
            epoch,
            scheduler=scheduler,
        )
        _, val_acc = evaluate_with_criterion(
            model, testloader, criterion_ls, device
        )
        print(f"[FT] Epoch {epoch}: Val Acc={val_acc:.2f}%")

        if val_acc > best_ft_acc:
            best_ft_acc = val_acc
            best_ft_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"[FT] New best accuracy: {best_ft_acc:.2f}% at epoch {epoch}")

    if best_ft_state is not None:
        torch.save(best_ft_state, cfg.final_pruned_ckpt)
        print(f"Saved final pruned+finetuned model with acc={best_ft_acc:.2f}%")
    else:
        torch.save(model.state_dict(), cfg.final_pruned_ckpt)
        print("Saved final pruned+finetuned model (no improvement recorded).")


if __name__ == "__main__":
    main()
