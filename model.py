"""Stateful multi-concept knowledge tracing with RCA, PTS, and IAW."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import ModelConfig
from data import ConceptRelations, Event


@dataclass
class RuntimeState:
    """Keep one learner's concept states and observed response history."""

    student_id: str | None = None
    concepts: dict[int, Tensor] = field(default_factory=dict)
    anchors: dict[int, Tensor] = field(default_factory=dict)
    history: list[Event] = field(default_factory=list)
    seen_events: int = 0
    closed_groups: set[str] = field(default_factory=set)
    seen_event_ids: set[str] = field(default_factory=set)
    last_position: int | None = None
    last_original_row: int | None = None

    def detach(self, history_limit: int | None = None) -> RuntimeState:
        """Cut recurrent gradients without replaying observed response updates."""
        history = self.history if history_limit is None else self.history[-history_limit:]
        if history_limit == 0:
            history = []
        return RuntimeState(
            student_id=self.student_id,
            concepts={key: value.detach() for key, value in self.concepts.items()},
            anchors={key: value.detach() for key, value in self.anchors.items()},
            history=list(history),
            seen_events=self.seen_events,
            closed_groups=set(self.closed_groups),
            seen_event_ids=set(self.seen_event_ids),
            last_position=self.last_position,
            last_original_row=self.last_original_row,
        )


class CausalBlock(nn.Module):
    """Apply post-normalized causal attention with differentiable matrix operations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.output = nn.Linear(config.dim, config.dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.dim),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm_attention = nn.LayerNorm(config.dim, eps=1e-5)
        self.norm_feedforward = nn.LayerNorm(config.dim, eps=1e-5)

    def forward(self, tokens: Tensor) -> Tensor:
        length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(1, 2, 0, 3).unbind(0)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = (query.float() @ key.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
            probabilities = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
            attended = self.dropout(probabilities) @ value.float()
        attended = attended.transpose(0, 1).reshape(length, width)
        tokens = self.norm_attention(tokens + self.dropout(self.output(attended)))
        return self.norm_feedforward(tokens + self.dropout(self.feedforward(tokens)))


@dataclass
class PredictionParts:
    long_logit: Tensor
    short_logit: Tensor
    pooled: Tensor
    item: Tensor
    context: Tensor
    states: Tensor
    anchors: Tensor

    @property
    def full_logit(self) -> Tensor:
        return self.long_logit + self.short_logit


class IASEKT(nn.Module):
    """Predict before feedback, then allocate feedback to current concept states."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.dim
        refinement_width = config.refinement_dim or width
        self.item_embedding = nn.Embedding(config.num_items, width)
        self.concept_embedding = nn.Embedding(config.num_concepts, width)
        self.response_embedding = nn.Embedding(2, width)
        self.position_embedding = nn.Embedding(config.max_history, width)
        self.initial_map = nn.Linear(width, width)
        self.encoder = nn.ModuleList(CausalBlock(config) for _ in range(config.layers))
        self.item_query = nn.Linear(width, width, bias=False)
        self.state_key = nn.Linear(width, width, bias=False)
        self.long_map = nn.Linear(3 * width, width, bias=False)
        self.long_readout = nn.Linear(width, 1, bias=False)
        self.item_bias = nn.Embedding(config.num_items, 1)
        self.short_map = nn.Linear(2 * width, width, bias=False)
        self.short_readout = nn.Linear(width, 1, bias=True)
        self.refinement_initial = nn.Linear(width, refinement_width, bias=False)
        self.refinement_map = nn.Sequential(
            nn.Linear(refinement_width + 1 + 2 * width, refinement_width),
            nn.GELU(),
            nn.Linear(refinement_width, refinement_width),
        )
        self.refinement_readout = nn.Linear(refinement_width, 1, bias=False)
        self.update_map = nn.Linear(3 * width + 1, width, bias=False)
        self.update_gate = nn.Linear(2 * width + 1, 1, bias=True)
        self.state_norm = nn.LayerNorm(width, eps=1e-5)
        nn.init.zeros_(self.item_bias.weight)

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    def detach_state(self, state: RuntimeState) -> RuntimeState:
        return state.detach(self.config.history_overlap)

    def _ids(self, values: Sequence[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _check_event(self, event: Event, state: RuntimeState, bind: bool = True) -> None:
        if state.student_id is not None and state.student_id != event.student_id:
            raise ValueError("Runtime state cannot be shared across learners")
        if not 0 <= event.item_id < self.config.num_items:
            raise ValueError("Item ID lies outside the model vocabulary")
        if not event.concept_ids or len(set(event.concept_ids)) != len(event.concept_ids):
            raise ValueError("Each response requires a nonempty set of distinct concepts")
        if any(not 0 <= key < self.config.num_concepts for key in event.concept_ids):
            raise ValueError("Concept ID lies outside the model vocabulary")
        if event.outcome not in (0, 1):
            raise ValueError("Outcomes must be binary")
        if bind and state.student_id is None:
            state.student_id = event.student_id

    def _ensure_concepts(self, event: Event, state: RuntimeState) -> None:
        for key in event.concept_ids:
            if key not in state.concepts:
                initial = torch.tanh(self.initial_map(self.concept_embedding(self._ids([key]))[0])).float()
                state.concepts[key] = initial
                state.anchors[key] = initial

    def _history_tokens(self, state: RuntimeState) -> Tensor:
        history = state.history[-self.config.max_history:]
        if not history:
            return self.item_embedding.weight.new_empty((0, self.config.dim))
        item_ids = self._ids([event.item_id for event in history])
        labels = self._ids([event.outcome for event in history])
        positions = torch.arange(len(history), device=self.device)
        concept_means = torch.stack([
            self.concept_embedding(self._ids(event.concept_ids)).mean(dim=0)
            for event in history
        ])
        return (
            self.item_embedding(item_ids)
            + self.response_embedding(labels)
            + self.position_embedding(positions)
            + concept_means
        ).float()

    def _long_prediction(self, item: Tensor, states: Tensor, context: Tensor, item_id: int) -> tuple[Tensor, Tensor]:
        query = self.item_query(item).float()
        keys = self.state_key(states).float()
        with torch.autocast(device_type=states.device.type, enabled=False):
            attention = torch.softmax((keys.float() @ query.float()) / math.sqrt(self.config.dim), dim=0)
            pooled = (attention[:, None] * states.float()).sum(dim=0)
        hidden = torch.tanh(self.long_map(torch.cat((pooled, item, context))))
        logit = self.long_readout(hidden).reshape(()) + self.item_bias(self._ids([item_id])).reshape(())
        return logit.float(), pooled

    def _prepare(self, event: Event, state: RuntimeState) -> PredictionParts:
        self._check_event(event, state)
        self._ensure_concepts(event, state)
        tokens = self._history_tokens(state)
        context = self.item_embedding.weight.new_zeros(self.config.dim)
        recent = context
        if len(tokens):
            encoded = tokens
            for layer in self.encoder:
                encoded = layer(encoded)
            context = encoded[-1].float()
            recent = tokens[-self.config.short_window:].mean(dim=0)
        item = self.item_embedding(self._ids([event.item_id]))[0].float()
        states = torch.stack([state.concepts[key] for key in event.concept_ids])
        anchors = torch.stack([state.anchors[key] for key in event.concept_ids])
        if torch.is_grad_enabled() and not states.requires_grad:
            states = states.detach().requires_grad_(True)
        long_logit, pooled = self._long_prediction(item, states, context, event.item_id)
        short_logit = self.short_readout(torch.tanh(self.short_map(torch.cat((recent, item))))).reshape(()).float()
        return PredictionParts(long_logit, short_logit, pooled, item, context, states, anchors)

    def predict(self, event: Event, state: RuntimeState | None = None) -> Tensor:
        """Return a probability without adding the current outcome to history."""
        state = RuntimeState() if state is None else state
        return torch.sigmoid(self._prepare(event, state).full_logit)

    @staticmethod
    def attenuation(short_logit: Tensor, temperature: float) -> Tensor:
        return torch.exp(-short_logit.abs() / temperature)

    @staticmethod
    def entropy_scale(entropy: Tensor, exponent: float) -> Tensor:
        """Use a finite zero-boundary derivative for positive entropy exponents."""
        if exponent == 0:
            return torch.ones_like(entropy)
        factor = 1.0 - entropy
        positive = factor > 0
        safe_factor = torch.where(positive, factor, torch.ones_like(factor))
        return torch.where(positive, safe_factor.pow(exponent), torch.zeros_like(factor))

    def _feedback(
        self, event: Event, state: RuntimeState, relations: ConceptRelations,
        prepared: PredictionParts | None = None,
    ) -> dict[str, Any]:
        with torch.enable_grad():
            parts = prepared if prepared is not None and prepared.states.requires_grad else self._prepare(event, state)
            label = parts.long_logit.new_tensor(float(event.outcome))
            likelihood = -F.binary_cross_entropy_with_logits(parts.long_logit, label)
            gradient = torch.autograd.grad(
                likelihood,
                parts.states,
                create_graph=self.training and self.config.differentiate_sensitivity,
                retain_graph=self.training,
            )[0]
            sensitivity = (gradient * (parts.states - parts.anchors)).sum(dim=-1).abs()
            base = torch.softmax(sensitivity / self.config.attribution_temperature, dim=0)
            count = len(event.concept_ids)
            if count == 1:
                entropy = base.new_zeros(())
            else:
                entropy = -(base * torch.log(base + self.config.epsilon)).sum() / math.log(count)
                entropy = entropy.clamp(0.0, 1.0)
            distinguishability = relations.distinguishability(event.concept_ids, device=self.device, dtype=base.dtype)
            score = entropy * (1.0 - distinguishability.mean())
            mode = self.config.branch_mode
            triggered = count > 1 and (
                mode == "refine_only" or (mode == "adaptive" and bool(score.detach() > self.config.refinement_threshold))
            )
            refined: Tensor | None = None
            differences_tensor: Tensor | None = None
            selected = base
            if triggered:
                differences = []
                for index in range(count):
                    copy = torch.cat((parts.states[:index], parts.anchors[index:index + 1], parts.states[index + 1:]), dim=0)
                    changed, _ = self._long_prediction(parts.item, copy, parts.context, event.item_id)
                    changed_likelihood = -F.binary_cross_entropy_with_logits(changed, label)
                    differences.append((likelihood - changed_likelihood).abs())
                differences_tensor = torch.stack(differences)
                representation = self.refinement_initial(parts.states)
                for _ in range(self.config.refinement_steps):
                    representation = self.refinement_map(torch.cat((
                        representation,
                        differences_tensor[:, None],
                        parts.pooled.expand(count, -1),
                        parts.item.expand(count, -1),
                    ), dim=-1))
                scores = self.refinement_readout(representation).reshape(-1).float() + differences_tensor
                refined = torch.softmax(scores / self.config.attribution_temperature, dim=0)
                selected = refined
            correction = self.attenuation(parts.short_logit, self.config.attenuation_temperature) * (
                label - torch.sigmoid(parts.long_logit)
            )
            coefficients = selected * distinguishability * self.entropy_scale(entropy, self.config.entropy_exponent)
            embeddings = self.concept_embedding(self._ids(event.concept_ids)).float()
            repeated_correction = correction.expand(count, 1)
            direction = torch.tanh(self.update_map(torch.cat((
                parts.states, parts.item.expand(count, -1), embeddings, repeated_correction,
            ), dim=-1)))
            gates = torch.sigmoid(self.update_gate(torch.cat((parts.states, direction, repeated_correction), dim=-1)))
            updated = self.state_norm(parts.states + coefficients[:, None] * gates * repeated_correction * direction)
            for index, key in enumerate(event.concept_ids):
                state.concepts[key] = updated[index]
            state.history.append(event)
            state.history = state.history[-self.config.max_history:]
            state.seen_events += 1
            auxiliary = {
                "base_weights": base,
                "refined_weights": refined,
                "coefficients": coefficients,
                "distinguishability": distinguishability,
                "triggered": triggered,
                "allocation_entropy": entropy,
                "refinement_score": score,
                "counterfactual_effects": differences_tensor,
                "selected_weights": selected,
                "correction": correction,
            }
            if not self.training:
                state.concepts = {key: value.detach() for key, value in state.concepts.items()}
                state.anchors = {key: value.detach() for key, value in state.anchors.items()}
                auxiliary = {key: value.detach() if isinstance(value, Tensor) else value for key, value in auxiliary.items()}
            return auxiliary

    def forward_sequence(
        self,
        events: Sequence[Event],
        relations: ConceptRelations,
        state: RuntimeState | None = None,
    ) -> dict[str, Any]:
        """Process chronological events with frozen pre-feedback bundle predictions."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Use torch.no_grad rather than torch.inference_mode for state sensitivity")
        state = RuntimeState() if state is None else state
        if relations.num_concepts != self.config.num_concepts:
            raise ValueError("Concept relations and model vocabulary sizes must agree")
        if events:
            owner = events[0].student_id
            if any(event.student_id != owner for event in events):
                raise ValueError("A sequence call must contain exactly one learner")
            if any(first.position > second.position for first, second in zip(events, events[1:])):
                raise ValueError("Sequence events must be supplied in chronological order")
            if len({event.event_id for event in events}) != len(events):
                raise ValueError("A sequence call cannot repeat a response event ID")
            if any(event.event_id in state.seen_event_ids for event in events):
                raise ValueError("An observed response cannot be replayed in the same learner trajectory")
            if state.last_position is not None:
                if events[0].position < state.last_position:
                    raise ValueError("Sequence calls must preserve chronological response order")
                if (events[0].position == state.last_position
                    and events[0].original_row is not None
                    and state.last_original_row is not None
                    and events[0].original_row < state.last_original_row):
                    raise ValueError("Equal-position events must preserve original row order")
            encountered_groups = set(state.closed_groups)
            active_group = None
            for event in events:
                self._check_event(event, state, bind=False)
                if event.group_id != active_group:
                    if event.group_id is not None:
                        if event.group_id in encountered_groups:
                            raise ValueError("A solving group must remain contiguous within one sequence call")
                        encountered_groups.add(event.group_id)
                    active_group = event.group_id
        outputs: dict[str, Any] = {key: [] for key in (
            "full_logits", "long_logits", "short_logits", "predictions", "labels",
            "base_weights", "refined_weights", "coefficients", "distinguishability",
            "triggered", "valid_mask", "allocation_entropy", "refinement_score",
            "counterfactual_effects", "selected_weights", "correction",
        )}
        cursor = 0
        while cursor < len(events):
            end = cursor + 1
            group_id = events[cursor].group_id
            if group_id is not None:
                while end < len(events) and events[end].group_id == group_id:
                    end += 1
            group = events[cursor:end]
            saved = [self._prepare(event, state) for event in group]
            first_index = state.seen_events
            updates: dict[int, dict[str, Any]] = {}
            order = sorted(range(len(group)), key=lambda index: (
                group[index].item_id,
                group[index].original_row if group[index].original_row is not None else index,
            )) if group_id is not None else [0]
            for index in order:
                prepared = saved[index] if group_id is None else None
                updates[index] = self._feedback(group[index], state, relations, prepared)
            for index, (event, parts) in enumerate(zip(group, saved)):
                long_logit, short_logit = parts.long_logit, parts.short_logit
                if not self.training:
                    long_logit, short_logit = long_logit.detach(), short_logit.detach()
                full_logit = long_logit + short_logit
                outputs["long_logits"].append(long_logit)
                outputs["short_logits"].append(short_logit)
                outputs["full_logits"].append(full_logit)
                outputs["predictions"].append(torch.sigmoid(full_logit))
                outputs["labels"].append(full_logit.new_tensor(float(event.outcome)))
                outputs["valid_mask"].append(first_index + index > 0)
                for key, value in updates[index].items():
                    outputs[key].append(value)
            if group_id is not None:
                state.closed_groups.add(group_id)
            state.seen_event_ids.update(event.event_id for event in group)
            state.last_position = group[-1].position
            state.last_original_row = group[-1].original_row
            cursor = end
        for key in ("long_logits", "short_logits", "full_logits", "predictions", "labels", "allocation_entropy", "refinement_score", "correction"):
            outputs[key] = torch.stack(outputs[key]) if outputs[key] else self.item_embedding.weight.new_empty((0,))
        for key in ("triggered", "valid_mask"):
            outputs[key] = torch.tensor(outputs[key], dtype=torch.bool, device=self.device)
        outputs["state"] = state
        return outputs

    def forward(self, events: Sequence[Event], relations: ConceptRelations, state: RuntimeState | None = None) -> dict[str, Any]:
        return self.forward_sequence(events, relations, state)
