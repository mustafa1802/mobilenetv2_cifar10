import torch
import torch.nn as nn

from model import channel_importance_l2


def layer_cap(name: str) -> float:
    """
    Per-layer max prune fraction based on MobileNetV2 block index.
    """
    if not name.startswith("features."):
        return 0.0

    parts = name.split(".")
    try:
        block_idx = int(parts[1])
    except (ValueError, IndexError):
        return 0.0

    if block_idx == 18:
        return 0.20  # final block

    if block_idx == 0:
        return 0.10

    if block_idx <= 3:
        return 0.10

    if block_idx <= 7:
        return 0.20

    if block_idx <= 13:
        return 0.30

    if block_idx <= 17:
        return 0.35

    return 0.0


def compute_prunability_scores(sensitivities, eps=1e-3, min_sens=1e-3):
    """
    Turn loss sensitivities into "prunability" scores (higher => more prunable).
    """
    scores = {}
    for name, s in sensitivities.items():
        s_clamped = max(s, min_sens)
        scores[name] = 1.0 / (eps + s_clamped)
    return scores


def allocate_prune_fractions(
    scores,
    param_counts,
    target_global_prune=0.25,
):
    """
    Compute per-layer prune fractions such that the approximate global
    parameter pruning fraction is `target_global_prune`, but respecting
    per-layer caps.
    """
    names = list(scores.keys())
    scores_vec = torch.tensor([scores[n] for n in names], dtype=torch.float32)
    params_vec = torch.tensor([param_counts[n] for n in names], dtype=torch.float32)

    mask = params_vec > 0
    if mask.sum() == 0:
        return {n: 0.0 for n in names}

    scores_vec = scores_vec[mask]
    params_vec = params_vec[mask]
    valid_names = [names[i] for i, m in enumerate(mask) if m.item()]

    score_sum = scores_vec.sum().item()
    if score_sum <= 0:
        base_frac = target_global_prune / float(len(valid_names))
        fractions = {n: base_frac for n in valid_names}
    else:
        rel_scores = scores_vec / score_sum
        target_params_to_remove = target_global_prune * params_vec.sum().item()
        layer_params_to_remove = target_params_to_remove * rel_scores
        fractions = (layer_params_to_remove / params_vec).tolist()
        fractions = {n: f for n, f in zip(valid_names, fractions)}

    final_fracs = {}
    for n in names:
        base_frac = max(0.0, fractions.get(n, 0.0))
        cap = layer_cap(n)
        final_fracs[n] = float(min(base_frac, cap))

    current_total = params_vec.sum().item()
    effective_removed = sum(
        param_counts[n] * final_fracs[n] for n in param_counts.keys()
    )
    effective_global_fraction = effective_removed / max(current_total, 1e-8)
    print(f"[allocate_prune_fractions] Effective global prune ≈ {effective_global_fraction:.3f}")
    return final_fracs


def apply_pruning_masks(model, prune_masks):
    """
    Zero out channels in-place using computed masks.
    prune_masks: dict {name: mask(out_channels)} for each conv name.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name in prune_masks:
                mask = prune_masks[name]
                W = module.weight.data
                if mask.shape[0] != W.size(0):
                    print(
                        f"[WARN] Mask size mismatch for layer {name}: "
                        f"mask={mask.shape[0]}, out_channels={W.size(0)}"
                    )
                    continue

                W_shape = W.shape
                W_flat = W.view(W.size(0), -1)
                W_flat[mask == 0, :] = 0.0
                module.weight.data = W_flat.view(W_shape)

                if module.bias is not None:
                    b = module.bias.data
                    b[mask == 0] = 0.0
                    module.bias.data = b


def apply_structured_channel_pruning(model, convs, prune_fracs):
    """
    Prune each Conv2d in convs by a fraction of its output channels according to prune_fracs.
    Returns (model, masks).
    """
    model.eval()

    importance_dict = {}
    for name, conv in convs:
        importance_dict[name] = channel_importance_l2(conv)

    masks = {}
    for name, conv in convs:
        frac = prune_fracs.get(name, 0.0)
        if frac <= 0.0:
            num_out = conv.weight.shape[0]
            masks[name] = torch.ones(num_out, dtype=torch.bool)
            continue

        imp = importance_dict[name]
        num_out = imp.shape[0]
        k = int(frac * num_out)
        if k <= 0:
            masks[name] = torch.ones(num_out, dtype=torch.bool)
            continue

        sorted_idx = torch.argsort(imp)
        prune_idx = sorted_idx[:k]

        mask = torch.ones(num_out, dtype=torch.bool)
        mask[prune_idx] = False
        masks[name] = mask

        print(
            f"[apply_structured_channel_pruning] Layer {name}: "
            f"prune {k}/{num_out} ({frac*100:.1f}%) channels."
        )

    apply_pruning_masks(model, masks)
    return model, masks
