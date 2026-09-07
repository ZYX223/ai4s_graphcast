"""One optimizer boundary, with lazy accumulation and manual DP over MP."""
from __future__ import annotations

import math

import torch

from .autoregressive_training import autoregressive_loss
from .training_recipe import clip_graphcast_gradients


def _require_all_finite(context, finite: bool, label: str) -> None:
    # Reduce the status, never a NaN scalar. Every rank takes the same branch.
    if context.global_sum(0. if finite else 1.) != 0:
        raise FloatingPointError(f"nonfinite {label} on a DP rank; update aborted")


def run_training_update(model, optimizer, microbatches, *, context,
                        checkpoint_segments=0, loss_grid_chunk_size=0):
    """Consume ``(AutoregressiveBatch, DP * local_valid / global_valid)`` lazily.

    Padding weights must be binary. The runtime supplies identical global
    microbatch boundaries on every rank, including locally all-padding slices.
    Returned diagnostics contain Python values only, averaged over global valid
    samples. This function never serializes or retains a rollout graph.
    """
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0., "lead_losses": [], "per_variable_lead_losses": {}}
    steps, valid_samples, scale_sum = 0, 0, 0.
    try:
        for batch, scale in microbatches:
            scale = float(scale)
            weights = batch.sample_weights
            binary = weights is None or bool(((weights == 0) | (weights == 1)).all())
            valid = int(batch.initial_input_features.shape[0]) if weights is None else int(weights.sum().item()) if binary else 0
            if context.global_sum(float(not binary or not math.isfinite(scale) or scale < 0)):
                raise ValueError("microbatch scales must be finite/non-negative and padding weights binary")
            prediction_finite = True
            try:
                result = autoregressive_loss(model, batch, checkpoint_segments=checkpoint_segments,
                                             loss_grid_chunk_size=loss_grid_chunk_size)
            except FloatingPointError:
                prediction_finite = False
            _require_all_finite(context, prediction_finite, "prediction")
            loss = result.loss * scale
            _require_all_finite(context, bool(torch.isfinite(loss)), "loss")
            loss.backward()
            totals["loss"] += float(loss.detach().cpu())
            leads = result.lead_losses.detach().cpu().tolist()
            if not totals["lead_losses"]:
                totals["lead_losses"] = [0.] * len(leads)
            for index, value in enumerate(leads):
                totals["lead_losses"][index] += value * scale
            for name, values in result.per_variable_lead_losses.items():
                target = totals["per_variable_lead_losses"].setdefault(name, [0.] * len(leads))
                for index, value in enumerate(values.detach().cpu().tolist()):
                    target[index] += value * scale
            steps += 1
            valid_samples += valid
            scale_sum += scale
            del loss, result, batch
        if not math.isclose(context.global_sum(scale_sum), context.dp_replicas, abs_tol=1e-6):
            raise ValueError("microbatch scales must sum globally to DP replicas")
        _require_all_finite(context, all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                                        if p.grad is not None), "gradients")
        synchronized = context.synchronize_gradients(model)
        _require_all_finite(context, all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                                        if p.grad is not None), "synchronized gradients")
        norm = clip_graphcast_gradients(model)
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    rows = context.gather_objects(totals)
    diagnostics = {
        "loss": sum(row["loss"] for row in rows) / context.dp_replicas,
        "lead_losses": [sum(row["lead_losses"][i] for row in rows) / context.dp_replicas
                        for i in range(len(totals["lead_losses"]))],
        "per_variable_lead_losses": {
            name: [sum(row["per_variable_lead_losses"][name][i] for row in rows) / context.dp_replicas
                   for i in range(len(values))]
            for name, values in totals["per_variable_lead_losses"].items()},
        "gradient_norm": norm, "finite": True, "accumulation_steps": steps,
        "effective_samples": int(context.global_sum(valid_samples)),
        "synchronized_parameters": synchronized,
    }
    return diagnostics
