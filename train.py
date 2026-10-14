"""Student-disjoint training, validation selection, and held-out assessment."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from config import ModelConfig, TrainingConfig
from data import ConceptRelations, build_cooccurrence, load_sequences, read_split
from evaluate import evaluate_model, iter_event_chunks, summarize_seeds
from losses import LossWeights, compute_objective
from model import IASEKT


def file_digest(path: Path) -> str:
    """Compute a content digest without loading the full event file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set independent numerical and learner-order random states."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def select_device(value: str = "auto") -> torch.device:
    """Resolve an available execution device and fail on invalid requests."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported execution devices are cpu and cuda.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_corpus(data_file: Path) -> tuple[list[Any], dict[str, Any], dict[str, list[Any]]]:
    vocabulary_path = data_file.parent / "vocab.json"
    split_path = data_file.parent / "split.json"
    if not vocabulary_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("The event directory must contain vocab.json and split.json.")
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    for count_key, values_key in (("num_items", "items"), ("num_concepts", "concepts")):
        count = vocabulary.get(count_key)
        values = vocabulary.get(values_key)
        if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
                or not isinstance(values, list) or len(values) != count
                or any(not isinstance(value, str) for value in values)
                or len(set(values)) != count):
            raise ValueError(f"The vocabulary requires {count} distinct string {values_key}.")
    sequences = load_sequences(data_file)
    for sequence in sequences:
        for event in sequence.events:
            if (event.item_id >= vocabulary["num_items"]
                    or any(concept >= vocabulary["num_concepts"] for concept in event.concept_ids)):
                raise ValueError("A canonical event lies outside its vocabulary.")
    partitions = read_split(sequences, split_path)
    for name in ("train", "validation", "test"):
        if not partitions[name]:
            raise ValueError(f"The {name} partition must contain at least one learner.")
    return sequences, vocabulary, partitions


def _precision_context(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    if precision != "bfloat16":
        raise ValueError("Precision must be float32 or bfloat16.")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bfloat16.")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def train_one_seed(
    data_file: Path, dataset: str, output_directory: Path,
    model_config: ModelConfig, training_config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Fit one seed and select the checkpoint using validation AUC only."""
    _, vocabulary, partitions = _load_corpus(data_file)
    split_metadata = json.loads((data_file.parent / "split.json").read_text(encoding="utf-8"))
    if split_metadata.get("dataset", dataset) != dataset:
        raise ValueError("The requested dataset name differs from the saved split metadata.")
    if (model_config.num_items != vocabulary["num_items"]
            or model_config.num_concepts != vocabulary["num_concepts"]):
        raise ValueError("Model vocabulary sizes differ from the canonical event vocabulary.")
    if not any(len(sequence.events) > 1 for sequence in partitions["train"]):
        raise ValueError("Training requires at least one learner with two or more events.")
    seed_everything(training_config.seed, training_config.deterministic)
    model = IASEKT(model_config).to(device)
    relations = build_cooccurrence(partitions["train"], model_config.num_concepts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate,
                                  weight_decay=0.0)
    objective_weights = LossWeights(
        separation=training_config.lambda_separation,
        consistency=training_config.lambda_consistency,
        structure=training_config.lambda_structure,
        l2=training_config.lambda_l2,
        short_magnitude=training_config.beta_magnitude,
        epsilon=model_config.epsilon,
    )
    chunk_count = sum(sum(1 for index, chunk in enumerate(iter_event_chunks(
        sequence.events, training_config.first_window, training_config.stride))
        if index > 0 or len(chunk) > 1) for sequence in partitions["train"])
    total_steps = math.ceil(chunk_count / training_config.batch_size) * training_config.epochs
    warmup_steps = math.ceil(total_steps * training_config.warmup_fraction)
    run_directory = output_directory / f"seed_{training_config.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best.pt"
    history_path = run_directory / "history.json"
    split_ids = {name: [sequence.student_id for sequence in values]
                 for name, values in partitions.items()}
    data_sha256 = file_digest(data_file)
    history: list[dict[str, Any]] = []
    best_auc = -math.inf
    epochs_without_improvement = 0
    optimizer_steps = 0
    order_rng = random.Random(training_config.seed)
    started = time.perf_counter()
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        learner_order = list(partitions["train"])
        order_rng.shuffle(learner_order)
        pending_outputs: list[dict[str, Any]] = []
        objective_totals: dict[str, float] = {}
        batch_count = 0

        def optimize_batch() -> None:
            nonlocal optimizer_steps, batch_count
            if not pending_outputs:
                return
            optimizer.zero_grad(set_to_none=True)
            losses = compute_objective(pending_outputs, model, objective_weights)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("The training objective became non-finite.")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.grad_clip, error_if_nonfinite=True)
            optimizer_steps += 1
            scale = min(1.0, optimizer_steps / warmup_steps) if warmup_steps else 1.0
            for group in optimizer.param_groups:
                group["lr"] = training_config.learning_rate * scale
            optimizer.step()
            for key, value in losses.items():
                objective_totals[key] = objective_totals.get(key, 0.0) + float(value.detach())
            objective_totals["gradient_norm"] = (objective_totals.get("gradient_norm", 0.0)
                                                 + float(gradient_norm))
            batch_count += 1
            pending_outputs.clear()

        for sequence in learner_order:
            state = None
            for chunk in iter_event_chunks(sequence.events, training_config.first_window,
                                           training_config.stride):
                with _precision_context(device, training_config.precision):
                    output = model.forward_sequence(chunk, relations, state=state)
                state = model.detach_state(output["state"])
                if bool(output["valid_mask"].any()):
                    pending_outputs.append(output)
                if len(pending_outputs) >= training_config.batch_size:
                    optimize_batch()
        optimize_batch()
        with _precision_context(device, training_config.precision):
            validation = evaluate_model(model, partitions["validation"], relations,
                                        training_config.first_window, training_config.stride)
        validation_auc = validation["auc"]
        if validation_auc is None:
            raise ValueError("Validation AUC is undefined; both outcome classes are required.")
        improved = float(validation_auc) > best_auc
        if improved:
            best_auc = float(validation_auc)
            epochs_without_improvement = 0
            payload = {
                "format_version": 1, "dataset": dataset,
                "model_config": model_config.to_dict(),
                "training_config": training_config.to_dict(),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "relations": relations.to_dict(), "vocabulary": vocabulary,
                "split_ids": split_ids, "data_sha256": data_sha256,
                "best_validation_auc": best_auc, "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "runtime": {"torch": str(torch.__version__), "device": str(device)},
            }
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
        epoch_record = {"epoch": epoch, "optimizer_steps": optimizer_steps,
                        "train_losses": {key: value / batch_count
                                         for key, value in objective_totals.items()},
                        "validation": validation, "improved": improved,
                        "elapsed_seconds": time.perf_counter() - started}
        history.append(epoch_record)
        _write_json(history_path, history)
        print(json.dumps({"seed": training_config.seed, "epoch": epoch,
                          "validation_auc": validation_auc, "best_validation_auc": best_auc,
                          "optimizer_steps": optimizer_steps}), flush=True)
        if epochs_without_improvement >= training_config.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    with _precision_context(device, training_config.precision):
        test = evaluate_model(model, partitions["test"], relations,
                              training_config.first_window, training_config.stride)
    result = {"seed": training_config.seed, "selected_epoch": checkpoint["epoch"],
              "best_validation_auc": best_auc, "checkpoint": str(checkpoint_path),
              "test": test, "elapsed_seconds": time.perf_counter() - started}
    _write_json(run_directory / "result.json", result)
    return result


def evaluate_checkpoint(checkpoint_path: Path, data_file: Path,
                        partition_name: str = "test", device_name: str = "auto") -> dict[str, Any]:
    """Evaluate stored learner identities with the checkpoint's training relations."""
    if partition_name not in ("validation", "test"):
        raise ValueError("Checkpoint assessment supports validation or test partitions.")
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if file_digest(data_file) != checkpoint["data_sha256"]:
        raise ValueError("The event file digest differs from the checkpoint's training corpus.")
    sequences, vocabulary, _ = _load_corpus(data_file)
    if vocabulary != checkpoint["vocabulary"]:
        raise ValueError("The event vocabulary differs from the checkpoint vocabulary.")
    by_student = {sequence.student_id: sequence for sequence in sequences}
    chosen_ids = checkpoint["split_ids"][partition_name]
    missing = set(chosen_ids) - set(by_student)
    if missing:
        raise ValueError(f"The event file lacks {len(missing)} checkpoint learners.")
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    training_config = TrainingConfig.from_dict(checkpoint["training_config"])
    model = IASEKT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    relations = ConceptRelations.from_dict(checkpoint["relations"])
    with _precision_context(device, training_config.precision):
        metrics = evaluate_model(model, [by_student[student_id] for student_id in chosen_ids],
                                 relations, training_config.first_window, training_config.stride)
    return {"partition": partition_name, "seed": training_config.seed,
            "selected_epoch": checkpoint["epoch"], "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IASE-KT with learner-disjoint splits.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2022, 2027)))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    parser.add_argument("--branch-mode", choices=("adaptive", "base_only", "refine_only"))
    arguments = parser.parse_args()
    _, vocabulary, _ = _load_corpus(arguments.data)
    model_config = (ModelConfig.load(arguments.model_config) if arguments.model_config
                    else ModelConfig(vocabulary["num_items"], vocabulary["num_concepts"]))
    training_config = (TrainingConfig.load(arguments.training_config) if arguments.training_config
                       else TrainingConfig())
    if arguments.branch_mode:
        model_config = replace(model_config, branch_mode=arguments.branch_mode)
    for field, value in (("epochs", arguments.epochs), ("batch_size", arguments.batch_size),
                         ("precision", arguments.precision)):
        if value is not None:
            training_config = replace(training_config, **{field: value})
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("Experiment seeds must be distinct.")
    device = select_device(arguments.device)
    results = [train_one_seed(arguments.data, arguments.dataset, arguments.output,
                              model_config, replace(training_config, seed=seed), device)
               for seed in arguments.seeds]
    summary = summarize_seeds([result["test"] for result in results])
    summary["dataset"] = arguments.dataset
    summary["seeds"] = arguments.seeds
    summary["individual_runs"] = results
    _write_json(arguments.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
