"""Masked objectives for item prediction and selective concept-state updates."""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    """Coefficients in the joint optimization objective."""

    separation: float = 0.03
    consistency: float = 0.1
    structure: float = 0.01
    l2: float = 1e-4
    short_magnitude: float = 0.1
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0
               for value in (self.separation, self.consistency, self.structure,
                             self.l2, self.short_magnitude)):
            raise ValueError("Loss coefficients must be finite and nonnegative.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")


def jensen_shannon(base: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Compare distributions while stopping gradients through the target."""
    if base.ndim != 1 or target.shape != base.shape or base.numel() == 0:
        raise ValueError("Concept distributions must be nonempty vectors of equal size.")
    p = base.float().clamp_min(epsilon)
    p = p / p.sum()
    q = target.detach().float().clamp_min(epsilon)
    q = q / q.sum()
    midpoint = (p + q) * 0.5
    return 0.5 * ((p * (p.log() - midpoint.log())).sum()
                  + (q * (q.log() - midpoint.log())).sum())


def compute_objective(
    outputs: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    model: nn.Module | None = None,
    weights: LossWeights | None = None,
) -> dict[str, Tensor]:
    """Apply the same valid-event mask to every event-dependent loss.

    Centered long- and short-term statistics span all supplied segments.
    Parameter regularization is explicit; an optimizer must therefore use
    zero decoupled weight decay when the L2 coefficient is nonzero.
    """
    settings = weights or LossWeights()
    batches = [outputs] if isinstance(outputs, Mapping) else list(outputs)
    if not batches:
        raise ValueError("At least one sequence output is required.")

    logits, labels, long_terms, short_terms = [], [], [], []
    consistency_terms, structural_terms = [], []
    anchor = batches[0]["long_logits"].sum() * 0.0
    valid_count = 0
    for batch in batches:
        long_logit = batch["long_logits"]
        short_logit = batch["short_logits"]
        outcome = batch["labels"]
        mask = batch["valid_mask"].to(dtype=torch.bool, device=long_logit.device)
        full_logit = batch.get("full_logits", long_logit + short_logit)
        if not (long_logit.ndim == 1 and long_logit.shape == short_logit.shape
                == outcome.shape == mask.shape == full_logit.shape):
            raise ValueError("Event logits, outcomes, and masks must share one vector shape.")
        logits.append(full_logit[mask].float())
        labels.append(outcome[mask].float())
        long_terms.append(long_logit[mask].float())
        short_terms.append(short_logit[mask].float())
        valid_count += int(mask.sum().item())
        flags = batch["triggered"].to(dtype=torch.bool, device=mask.device)
        if flags.shape != mask.shape:
            raise ValueError("Refinement flags must align with event masks.")
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            coefficient = batch["coefficients"][index]
            distinguishability = batch["distinguishability"][index]
            if coefficient.shape != distinguishability.shape:
                raise ValueError("Update and structural vectors must align.")
            structural_terms.append(((1.0 - distinguishability.float())
                                     * coefficient.float().square()).sum())
            if bool(flags[index].item()):
                refined = batch["refined_weights"][index]
                if refined is None:
                    raise ValueError("A refined response requires its concept distribution.")
                consistency_terms.append(jensen_shannon(
                    batch["base_weights"][index], refined, settings.epsilon))

    if valid_count:
        full = torch.cat(logits)
        truth = torch.cat(labels)
        z_long = torch.cat(long_terms)
        z_short = torch.cat(short_terms)
        prediction = F.binary_cross_entropy_with_logits(full, truth)
        centered_long = z_long - z_long.mean()
        centered_short = z_short - z_short.mean()
        denominator = (centered_long.norm() * centered_short.norm()
                       + settings.epsilon)
        correlation = (centered_long.dot(centered_short) / denominator).square()
        magnitude = z_short.square().mean()
        separation = correlation + settings.short_magnitude * magnitude
        structure = torch.stack(structural_terms).sum() / valid_count
    else:
        prediction = correlation = magnitude = separation = structure = anchor
    consistency = (torch.stack(consistency_terms).sum()
                   / (len(consistency_terms) + settings.epsilon)
                   if consistency_terms else anchor)
    if model is None:
        if settings.l2:
            raise ValueError("Explicit L2 regularization requires the model parameters.")
        regularization = anchor
    else:
        regularization = anchor
        for parameter in model.parameters():
            if parameter.requires_grad:
                regularization = regularization + parameter.square().sum()
    total = (prediction + settings.separation * separation
             + settings.consistency * consistency + settings.structure * structure
             + settings.l2 * regularization)
    return {"total": total, "prediction": prediction, "separation": separation,
            "correlation": correlation, "short_magnitude": magnitude,
            "consistency": consistency, "structure": structure,
            "l2": regularization,
            "valid_events": anchor.new_tensor(float(valid_count)),
            "refined_events": anchor.new_tensor(float(len(consistency_terms)))}
