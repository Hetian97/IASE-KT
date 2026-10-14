"""Online correctness metrics and matched subsequent-concept evaluation."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch


def iter_event_chunks(
    events: Sequence[Any], first_window: int = 200, stride: int = 100,
) -> Iterator[Sequence[Any]]:
    """Emit each event once while keeping consecutive solving groups intact."""
    if first_window < 1 or stride < 1:
        raise ValueError("Window and stride sizes must be positive.")
    start = 0
    limit = first_window
    while start < len(events):
        end = start
        while end < len(events):
            group_end = end + 1
            group_id = getattr(events[end], "group_id", None)
            if group_id is not None:
                while (group_end < len(events)
                       and getattr(events[group_end], "group_id", None) == group_id):
                    group_end += 1
            if end > start and group_end - start > limit:
                break
            end = group_end
            if end - start >= limit:
                break
        yield events[start:end]
        start, limit = end, stride


def binary_metrics(labels: Iterable[float], scores: Iterable[float]) -> dict[str, Any]:
    """Compute ROC AUC with average ranks, including tied predictions."""
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("Labels and scores must be equally sized vectors.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Labels and predictions must be finite.")
    if not np.isin(y, [0.0, 1.0]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Labels must be binary and probabilities must lie in [0, 1].")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if not y.size:
        return {"auc": None, "accuracy": None, "log_loss": None, "events": 0,
                "positives": 0, "negatives": 0, "auc_status": "No scored events."}
    accuracy = float(((p >= 0.5) == y).mean())
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    log_loss = float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean())
    auc = None
    if positives and negatives:
        order = np.argsort(p, kind="mergesort")
        sorted_scores = p[order]
        ranks = np.empty(y.size, dtype=np.float64)
        left = 0
        while left < y.size:
            right = left + 1
            while right < y.size and sorted_scores[right] == sorted_scores[left]:
                right += 1
            ranks[order[left:right]] = 0.5 * (left + 1 + right)
            left = right
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                    / (positives * negatives))
    return {"auc": auc, "accuracy": accuracy, "log_loss": log_loss,
            "events": int(y.size), "positives": positives, "negatives": negatives,
            "auc_status": "Defined." if auc is not None
            else "ROC AUC requires both outcome classes."}


def evaluate_model(
    model: torch.nn.Module, sequences: Iterable[Any], relations: Any,
    first_window: int = 200, stride: int = 100, fc_horizon: int = 20,
) -> dict[str, Any]:
    """Score saved predictions before labels enter each learner's state."""
    if fc_horizon < 1:
        raise ValueError("The concept-follow-up horizon must be positive.")
    was_training = model.training
    model.eval()
    labels, predictions, fc_labels, fc_predictions = [], [], [], []
    refined = multi_concept = 0
    try:
        with torch.no_grad():
            for sequence in sequences:
                state = None
                learner_scores, learner_valid = [], []
                for chunk in iter_event_chunks(sequence.events, first_window, stride):
                    output = model.forward_sequence(chunk, relations, state=state)
                    state = model.detach_state(output["state"])
                    values = output["predictions"].detach().cpu().tolist()
                    mask = output["valid_mask"].detach().cpu().tolist()
                    flags = output["triggered"].detach().cpu().tolist()
                    truth = output["labels"].detach().cpu().tolist()
                    learner_scores.extend(values)
                    learner_valid.extend(mask)
                    for event, score, label, valid, flag in zip(chunk, values, truth, mask, flags):
                        if valid:
                            labels.append(label)
                            predictions.append(score)
                            refined += int(flag)
                            multi_concept += int(len(event.concept_ids) > 1)
                events = sequence.events
                for index, event in enumerate(events):
                    if not learner_valid[index] or len(event.concept_ids) <= 1:
                        continue
                    for concept in event.concept_ids:
                        valid_steps = 0
                        for future in range(index + 1, len(events)):
                            if not learner_valid[future]:
                                continue
                            valid_steps += 1
                            if valid_steps > fc_horizon:
                                break
                            successor = events[future]
                            if len(successor.concept_ids) == 1 and successor.concept_ids[0] == concept:
                                fc_labels.append(float(successor.outcome))
                                fc_predictions.append(learner_scores[future])
                                break
    finally:
        model.train(was_training)
    metrics = binary_metrics(labels, predictions)
    metrics["fc_metrics"] = binary_metrics(fc_labels, fc_predictions)
    metrics["refined_events"] = refined
    metrics["multi_concept_events"] = multi_concept
    metrics["refinement_rate"] = refined / multi_concept if multi_concept else 0.0
    metrics["fc_horizon"] = fc_horizon
    return metrics


def summarize_seeds(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report sample standard deviations without imputing undefined AUCs."""
    summary: dict[str, Any] = {"runs": len(results), "metrics": {}}
    for key in ("auc", "accuracy", "log_loss", "fc_auc"):
        observations = [result.get("fc_metrics", {}).get("auc") if key == "fc_auc"
                        else result.get(key) for result in results]
        values = [float(value) for value in observations
                  if value is not None and math.isfinite(float(value))]
        summary["metrics"][key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "defined_runs": len(values),
        }
    return summary


def paired_comparison(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    """Compare matched seed outcomes with an optional paired t-test."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("A paired comparison requires at least two matched runs.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired outcomes must be finite.")
    difference = a - b
    result: dict[str, Any] = {"pairs": int(a.size), "mean_difference": float(difference.mean()),
                              "difference_std": float(difference.std(ddof=1)), "p_value": None}
    if float(difference.std(ddof=1)) == 0:
        result["status"] = "The paired t-test is undefined for a constant difference."
        return result
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        result["status"] = "Install scipy to compute the paired t-test."
        return result
    result["p_value"] = float(ttest_rel(a, b).pvalue)
    result["status"] = "Two-sided paired t-test; multiplicity correction is separate."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an IASE-KT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    from train import evaluate_checkpoint
    metrics = evaluate_checkpoint(arguments.checkpoint, arguments.data,
                                  arguments.split, arguments.device)
    content = json.dumps(metrics, indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
