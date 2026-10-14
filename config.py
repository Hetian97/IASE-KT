"""Validated configurations for IASE-KT experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, TypeVar


ConfigT = TypeVar("ConfigT", bound="SerializableConfig")


class SerializableConfig:
    """Persist configuration values without silently accepting unknown fields."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        return cls(**values)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls: type[ConfigT], path: str | Path) -> ConfigT:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite(name: str, value: float, lower: float = 0.0, inclusive: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or (value < lower if inclusive else value <= lower):
        comparison = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be finite and {comparison} {lower}")


@dataclass(frozen=True)
class ModelConfig(SerializableConfig):
    num_items: int
    num_concepts: int
    dim: int = 128
    heads: int = 4
    layers: int = 2
    ffn_dim: int = 512
    dropout: float = 0.1
    max_history: int = 200
    short_window: int = 20
    attribution_temperature: float = 0.7
    attenuation_temperature: float = 1.0
    refinement_threshold: float = 0.55
    refinement_steps: int = 2
    entropy_exponent: float = 1.0
    epsilon: float = 1e-8
    branch_mode: str = "adaptive"
    differentiate_sensitivity: bool = True
    history_overlap: int | None = None
    refinement_dim: int | None = None

    def __post_init__(self) -> None:
        for name in ("num_items", "num_concepts", "dim", "heads", "layers", "ffn_dim",
                     "max_history", "short_window", "refinement_steps"):
            _positive_integer(name, getattr(self, name))
        if self.refinement_dim is None:
            object.__setattr__(self, "refinement_dim", self.dim)
        _positive_integer("refinement_dim", self.refinement_dim)
        if self.history_overlap is None:
            object.__setattr__(self, "history_overlap", min(100, self.max_history // 2))
        if isinstance(self.history_overlap, bool) or not isinstance(self.history_overlap, int) or not 0 <= self.history_overlap < self.max_history:
            raise ValueError("history_overlap must lie in [0, max_history)")
        if self.branch_mode not in {"adaptive", "base_only", "refine_only"}:
            raise ValueError("branch_mode must be adaptive, base_only, or refine_only")
        if not isinstance(self.differentiate_sensitivity, bool):
            raise ValueError("differentiate_sensitivity must be boolean")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.short_window > self.max_history:
            raise ValueError("short_window cannot exceed max_history")
        _finite("dropout", self.dropout, inclusive=True)
        if self.dropout >= 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        _finite("refinement_threshold", self.refinement_threshold, inclusive=True)
        if self.refinement_threshold > 1.0:
            raise ValueError("refinement_threshold must lie in [0, 1]")
        for name in ("attribution_temperature", "attenuation_temperature", "epsilon"):
            _finite(name, getattr(self, name))
        _finite("entropy_exponent", self.entropy_exponent, inclusive=True)


@dataclass(frozen=True)
class TrainingConfig(SerializableConfig):
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 100
    patience: int = 10
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    seed: int = 2022
    first_window: int = 200
    stride: int = 100
    lambda_separation: float = 0.03
    lambda_consistency: float = 0.1
    lambda_structure: float = 0.01
    lambda_l2: float = 1e-4
    beta_magnitude: float = 0.1
    precision: str = "float32"
    deterministic: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "patience", "first_window", "stride"):
            _positive_integer(name, getattr(self, name))
        if self.stride > self.first_window:
            raise ValueError("stride cannot exceed first_window")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "grad_clip"):
            _finite(name, getattr(self, name))
        for name in ("warmup_fraction", "lambda_separation", "lambda_consistency",
                     "lambda_structure", "lambda_l2", "beta_magnitude"):
            _finite(name, getattr(self, name), inclusive=True)
        if self.warmup_fraction >= 1.0:
            raise ValueError("warmup_fraction must lie in [0, 1)")
