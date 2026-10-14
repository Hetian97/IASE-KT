"""Executable checks for mathematical, temporal, and experiment integrity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from config import ModelConfig, TrainingConfig
from data import (ConceptRelations, Event, StudentSequence, build_cooccurrence, import_ednet, import_assist,
                  load_sequences, prepare_dataset, read_split, split_students, write_sequences)
from evaluate import binary_metrics, evaluate_model, iter_event_chunks, summarize_seeds
from losses import LossWeights, compute_objective, jensen_shannon
from model import IASEKT, RuntimeState


def small_config(**changes) -> ModelConfig:
    values = dict(num_items=4, num_concepts=3, dim=8, heads=2, layers=1,
                  ffn_dim=16, dropout=0.0, max_history=8, short_window=2,
                  history_overlap=4)
    values.update(changes)
    return ModelConfig(**values)


def learner(student: str = "s0", length: int = 12, offset: int = 0) -> StudentSequence:
    assignments = ((0,), (1,), (0, 1), (2,))
    return StudentSequence(student, tuple(
        Event(student, position % 4, assignments[position % 4],
              (position + offset) % 2, position, f"e{position}")
        for position in range(length)
    ))


def fixed_model(**changes) -> IASEKT:
    torch.manual_seed(17)
    return IASEKT(small_config(**changes))


class DataChecks(unittest.TestCase):
    def test_configuration_roundtrip(self):
        config = small_config(branch_mode="refine_only")
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)
        self.assertEqual(TrainingConfig.from_dict(TrainingConfig().to_dict()), TrainingConfig())

    def test_invalid_configuration(self):
        for changes in (dict(dim=7), dict(dropout=1), dict(attribution_temperature=0),
                        dict(history_overlap=8), dict(branch_mode="other")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                small_config(**changes)

    def test_unknown_configuration_field(self):
        with self.assertRaises(ValueError):
            ModelConfig.from_dict(dict(num_items=4, num_concepts=3, unused=1))

    def test_split_is_disjoint_and_order_invariant(self):
        sequences = [learner(f"s{i}") for i in range(12)]
        first = split_students(sequences, "integrity", counts=(8, 2, 2))
        second = split_students(list(reversed(sequences)), "integrity", counts=(8, 2, 2))
        self.assertEqual({k: [x.student_id for x in v] for k, v in first.items()},
                         {k: [x.student_id for x in v] for k, v in second.items()})
        identities = [x.student_id for partition in first.values() for x in partition]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(set(identities), {x.student_id for x in sequences})

    def test_sampling_matches_independent_hash_order(self):
        sequences = [learner(f"s{i}") for i in range(12)]
        expected = sorted(sequences, key=lambda seq: hashlib.sha256(
            f"IASEKT-v1|sample|integrity|{seq.student_id}".encode()).digest())[:6]
        partitions = split_students(sequences, "integrity", counts=(4, 1, 1), sample_size=6)
        self.assertEqual({x.student_id for rows in partitions.values() for x in rows},
                         {x.student_id for x in expected})

    def test_skewed_ratios_keep_nonempty_partitions(self):
        for count in range(3, 13):
            partitions = split_students([learner(f"s{i}") for i in range(count)],
                                        "integrity", ratios=(98, 1, 1))
            self.assertTrue(all(partitions.values()))
            self.assertEqual(sum(map(len, partitions.values())), count)

    def test_complete_response_merge_preserves_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(
                "student_id,item_id,concept_ids,outcome,position,event_id,group_id\n"
                "s,2,0,1,0,e0,\n"
                "s,2,1,1,0,e0,\n"
                "s,2,0|1,1,1,e1,\n", encoding="utf-8")
            events = load_sequences(path)[0].events
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].concept_ids, (0, 1))
            self.assertEqual(events[1].event_id, "e1")

    def test_conflicting_event_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(
                "student_id,item_id,concept_ids,outcome,position,event_id\n"
                "s,2,0,1,0,e0\ns,2,1,0,0,e0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_sequences(path)

    def test_csv_and_split_roundtrip(self):
        sequences = [learner(f"s{i}") for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            prepare_dataset(sequences, {"num_items": 4, "num_concepts": 3},
                            directory, "integrity", counts=(4, 1, 1))
            loaded = load_sequences(Path(directory) / "canonical.csv")
            partitions = read_split(loaded, Path(directory) / "split.json")
            self.assertEqual(sorted(loaded, key=lambda x: x.student_id), sequences)
            self.assertEqual([len(partitions[key]) for key in ("train", "validation", "test")], [4, 1, 1])

    def test_unique_item_jaccard_ignores_response_frequency(self):
        sequence = learner()
        once = build_cooccurrence([sequence], 3)
        repeated = build_cooccurrence([sequence, learner("s1", 40)], 3)
        self.assertEqual(once.to_dict(), repeated.to_dict())
        self.assertAlmostEqual(once.overlap(0, 1), 1 / (3 + 1e-8))
        self.assertEqual(once.overlap(0, 2), 0)

    def test_relations_use_only_supplied_training_items(self):
        training = StudentSequence("s", (Event("s", 0, (0,), 1, 0, "a"),))
        relations = build_cooccurrence([training], 3)
        self.assertEqual(relations.item_counts, (1, 0, 0))
        restored = ConceptRelations.from_dict(relations.to_dict())
        self.assertEqual(restored.to_dict(), relations.to_dict())

    def test_single_concept_distinguishability(self):
        relations = build_cooccurrence([learner()], 3)
        self.assertEqual(float(relations.distinguishability((0,))[0]), 1.0)

    def test_original_row_survives_csv_roundtrip(self):
        event = Event("s", 0, (0,), 1, 0, "e", "g", original_row=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            write_sequences([StudentSequence("s", (event,))], path)
            self.assertEqual(load_sequences(path)[0].events[0].original_row, 9)

    def test_split_students_require_explicit_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"students": {"train": "s0", "validation": [], "test": []}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_split([learner()], path)

    def test_native_ednet_preserves_bundle_order_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "KT1").mkdir()
            (root / "questions.csv").write_text(
                "question_id,correct_answer,tags\nq1,a,0;1\nq2,b,1\nq3,c,2\n", encoding="utf-8")
            (root / "KT1" / "u1.csv").write_text(
                "timestamp,solving_id,question_id,user_answer\n"
                "20,1,q1,a\n10,1,q1,c\n30,2,q2,b\n40,3,q3,c\n", encoding="utf-8")
            sequences, vocabulary = import_ednet(root / "KT1", root / "questions.csv")
            events = sequences[0].events
            self.assertEqual([event.original_row for event in events], [1, 0, 2, 3])
            config = small_config(num_items=vocabulary["num_items"], num_concepts=vocabulary["num_concepts"])
            model = IASEKT(config).eval()
            output = model(events[:2], build_cooccurrence(sequences, config.num_concepts))
            self.assertEqual([event.original_row for event in output["state"].history], [0, 1])
            torch.testing.assert_close(output["predictions"][0], output["predictions"][1], rtol=0, atol=0)

    def test_native_assist_merges_concepts_without_merging_repeated_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assist.csv"
            path.write_text(
                "user_id,problem_id,skill_id,correct,order_id\n"
                "s,q1,0,1,2\ns,q1,1,1,2\ns,q1,0,0,3\n"
                "s,q1,1,0,3\ns,q2,1,1,1\ns,q3,2,1,4\n", encoding="utf-8")
            sequences, _ = import_assist(path)
            events = sequences[0].events
            self.assertEqual(len(events), 4)
            self.assertEqual([event.original_row for event in events], [4, 0, 2, 5])
            self.assertEqual(events[1].concept_ids, events[2].concept_ids)
            self.assertNotEqual(events[1].event_id, events[2].event_id)

    def test_direct_sequence_rejects_original_row_inversion_at_equal_position(self):
        events = (Event("s", 0, (0,), 1, 0, "a", original_row=5),
                  Event("s", 0, (0,), 0, 0, "b", original_row=1))
        with self.assertRaises(ValueError):
            StudentSequence("s", events)


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.sequence = learner()
        self.relations = build_cooccurrence([self.sequence], 3)

    def test_current_and_future_labels_do_not_change_saved_prediction(self):
        model = fixed_model().eval()
        for index in range(len(self.sequence.events)):
            altered = tuple(replace(event, outcome=1 - event.outcome) if position >= index else event
                            for position, event in enumerate(self.sequence.events))
            with torch.no_grad():
                original = model(self.sequence.events, self.relations)["predictions"]
                changed = model(altered, self.relations)["predictions"]
            torch.testing.assert_close(original[:index + 1], changed[:index + 1], rtol=0, atol=0)

    def test_bundle_predictions_are_frozen_before_feedback(self):
        model = fixed_model().eval()
        events = list(self.sequence.events[:6])
        events[1] = replace(events[1], group_id="bundle")
        events[2] = replace(events[2], group_id="bundle")
        changed = list(events)
        changed[1] = replace(changed[1], outcome=1 - changed[1].outcome)
        with torch.no_grad():
            first = model(events, self.relations)["predictions"]
            second = model(changed, self.relations)["predictions"]
        torch.testing.assert_close(first[:3], second[:3], rtol=0, atol=0)

    def test_split_bundle_is_rejected_even_after_history_is_cleared(self):
        model = fixed_model(history_overlap=0).eval()
        first = replace(self.sequence.events[0], group_id="group")
        second = replace(self.sequence.events[1], group_id="group")
        state = model.detach_state(model([first], self.relations)["state"])
        self.assertEqual(state.history, [])
        before = state.seen_events
        with self.assertRaises(ValueError):
            model([second], self.relations, state)
        self.assertEqual(state.seen_events, before)
        self.assertEqual(state.closed_groups, {"group"})

    def test_noncontiguous_group_is_rejected_before_any_state_update(self):
        model = fixed_model().eval()
        events = list(self.sequence.events[:3])
        events[0] = replace(events[0], group_id="group")
        events[2] = replace(events[2], group_id="group")
        state = RuntimeState()
        with self.assertRaises(ValueError):
            model(events, self.relations, state)
        self.assertEqual(state.seen_events, 0)
        self.assertIsNone(state.student_id)

    def test_response_replay_and_backward_time_are_rejected_across_calls(self):
        model = fixed_model(history_overlap=0).eval()
        output = model(self.sequence.events[:4], self.relations)
        state = model.detach_state(output["state"])
        self.assertEqual(state.history, [])
        with self.assertRaises(ValueError):
            model(self.sequence.events[:1], self.relations, state)
        backward = replace(self.sequence.events[1], event_id="new")
        with self.assertRaises(ValueError):
            model([backward], self.relations, state)
        self.assertEqual(state.seen_events, 4)
        self.assertEqual(len(state.seen_event_ids), 4)

    def test_recurrent_gradients_stop_only_at_segment_boundary(self):
        model = fixed_model(branch_mode="refine_only")
        first = model(self.sequence.events[:2], self.relations)
        previous = first["state"].concepts[0]
        later = model(self.sequence.events[2:3], self.relations, first["state"])
        gradient = torch.autograd.grad(later["full_logits"].sum(), previous, retain_graph=True, allow_unused=True)[0]
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        detached = model.detach_state(later["state"])
        final = model(self.sequence.events[3:4], self.relations, detached)
        blocked = torch.autograd.grad(final["full_logits"].sum(), previous, retain_graph=True, allow_unused=True)[0]
        self.assertIsNone(blocked)

    def test_student_state_isolation(self):
        model = fixed_model().eval()
        output = model(self.sequence.events[:2], self.relations)
        with self.assertRaises(ValueError):
            model(learner("other").events[:1], self.relations, output["state"])

    def test_invalid_sequence_does_not_mutate_runtime_state(self):
        model = fixed_model().eval()
        state = RuntimeState()
        bad = Event("s0", 8, (0,), 0, 1, "bad")
        with self.assertRaises(ValueError):
            model([self.sequence.events[0], bad], self.relations, state)
        self.assertIsNone(state.student_id)
        self.assertEqual(state.seen_events, 0)
        self.assertEqual(state.concepts, {})
        self.assertEqual(state.history, [])
        with self.assertRaises(ValueError):
            model.predict(bad, state)
        self.assertIsNone(state.student_id)

    def test_attention_statistics_remain_float32_under_autocast(self):
        observed = []
        original = torch.softmax
        def record(value, *args, **kwargs):
            observed.append(value.dtype)
            return original(value, *args, **kwargs)
        model = fixed_model().eval()
        torch.softmax = record
        try:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                model(self.sequence.events[:6], self.relations)
        finally:
            torch.softmax = original
        self.assertTrue(observed)
        self.assertTrue(all(dtype == torch.float32 for dtype in observed))

    def test_unrelated_concept_states_are_unchanged(self):
        model = fixed_model().eval()
        output = model(self.sequence.events[:4], self.relations)
        state = output["state"]
        untouched = state.concepts[2].clone()
        next_event = Event("s0", 0, (0,), 0, 4, "next")
        model([next_event], self.relations, state)
        torch.testing.assert_close(untouched, state.concepts[2], rtol=0, atol=0)

    def test_attenuation_is_symmetric_and_monotone(self):
        values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        factors = IASEKT.attenuation(values, 1.0)
        torch.testing.assert_close(factors, torch.flip(factors, dims=(0,)))
        self.assertLess(float(factors[0]), float(factors[1]))
        self.assertEqual(float(factors[2]), 1.0)

    def test_fractional_entropy_exponents_have_finite_boundary_gradients(self):
        events = tuple(replace(event, position=index, event_id=f"r{index}")
                       for index, event in enumerate((self.sequence.events[2], self.sequence.events[0],
                                                      self.sequence.events[1], self.sequence.events[2])))
        for exponent in (0.0, 0.2, 0.5, 1.0, 2.0):
            with self.subTest(exponent=exponent):
                model = fixed_model(entropy_exponent=exponent, branch_mode="refine_only")
                compute_objective(model(events, self.relations), model)["total"].backward()
                self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                    for parameter in model.parameters()))

    def test_bundle_updates_follow_item_and_original_row(self):
        model = fixed_model().eval()
        events = (
            Event("s0", 2, (0, 1), 1, 0, "a", "g", original_row=9),
            Event("s0", 1, (1,), 0, 1, "b", "g", original_row=7),
            Event("s0", 1, (1,), 1, 2, "c", "g", original_row=2),
        )
        state = model(events, self.relations)["state"]
        self.assertEqual([event.event_id for event in state.history], ["c", "b", "a"])

    def test_cpu_bfloat16_forward_and_backward(self):
        model = fixed_model(branch_mode="refine_only")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(self.sequence.events[:8], self.relations)
        compute_objective(output, model)["total"].backward()
        self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()))

    def test_branch_endpoints_and_strict_threshold(self):
        for mode in ("base_only", "refine_only"):
            model = fixed_model(branch_mode=mode).eval()
            output = model(self.sequence.events, self.relations)
            expected = [mode == "refine_only" and len(event.concept_ids) > 1 for event in self.sequence.events]
            self.assertEqual(output["triggered"].tolist(), expected)
        output = fixed_model(refinement_threshold=1.0).eval()(self.sequence.events, self.relations)
        self.assertFalse(bool(output["triggered"].any()))

    def test_counterfactual_resets_are_independent(self):
        model = fixed_model(branch_mode="refine_only").eval()
        state = model(self.sequence.events[:2], self.relations)["state"]
        event = self.sequence.events[2]
        with torch.enable_grad():
            parts = model._prepare(event, state)
            label = parts.long_logit.new_tensor(float(event.outcome))
            baseline = -F.binary_cross_entropy_with_logits(parts.long_logit, label)
            expected = []
            for index in range(len(event.concept_ids)):
                states = parts.states.clone()
                states[index] = parts.anchors[index]
                logit, _ = model._long_prediction(parts.item, states, parts.context, event.item_id)
                expected.append((baseline + F.binary_cross_entropy_with_logits(logit, label)).abs())
        output = model([event], self.relations, state)
        torch.testing.assert_close(output["counterfactual_effects"][0], torch.stack(expected).detach())

    def test_sensitivity_matches_finite_difference(self):
        model = fixed_model(branch_mode="base_only").eval()
        state = model(self.sequence.events[:2], self.relations)["state"]
        event = self.sequence.events[2]
        parts = model._prepare(event, state)
        label = parts.long_logit.new_tensor(float(event.outcome))
        step = 1e-3
        gradient = torch.zeros_like(parts.states)
        with torch.no_grad():
            for row in range(parts.states.shape[0]):
                for column in range(parts.states.shape[1]):
                    plus, minus = parts.states.clone(), parts.states.clone()
                    plus[row, column] += step
                    minus[row, column] -= step
                    positive, _ = model._long_prediction(parts.item, plus, parts.context, event.item_id)
                    negative, _ = model._long_prediction(parts.item, minus, parts.context, event.item_id)
                    gradient[row, column] = (-F.binary_cross_entropy_with_logits(positive, label)
                                             + F.binary_cross_entropy_with_logits(negative, label)) / (2 * step)
            magnitude = (gradient * (parts.states - parts.anchors)).sum(-1).abs()
            expected = torch.softmax(magnitude / model.config.attribution_temperature, dim=0)
        actual = model([event], self.relations, state)["base_weights"][0]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_feedback_uses_saved_logits_with_dropout(self):
        model = fixed_model(dropout=0.4)
        output = model(self.sequence.events[:6], self.relations)
        expected = IASEKT.attenuation(output["short_logits"], 1.0) * (
            output["labels"] - torch.sigmoid(output["long_logits"]))
        torch.testing.assert_close(output["correction"], expected, rtol=0, atol=0)

    def test_first_event_mask_and_window_continuity(self):
        model = fixed_model().eval()
        state, scored = None, 0
        for chunk in iter_event_chunks(self.sequence.events, 8, 4):
            output = model(chunk, self.relations, state)
            scored += int(output["valid_mask"].sum())
            state = model.detach_state(output["state"])
            self.assertLessEqual(len(state.history), 4)
            self.assertTrue(all(not value.requires_grad for value in state.concepts.values()))
        self.assertEqual(scored, len(self.sequence.events) - 1)
        self.assertEqual(state.seen_events, len(self.sequence.events))

    def test_prediction_is_logit_sum(self):
        output = fixed_model().eval()(self.sequence.events[:6], self.relations)
        torch.testing.assert_close(output["predictions"], torch.sigmoid(output["long_logits"] + output["short_logits"]))

    def test_training_gradients_are_finite_and_reach_refinement(self):
        model = fixed_model(branch_mode="refine_only")
        output = model(self.sequence.events, self.relations)
        compute_objective(output, model)["total"].backward()
        self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()))
        self.assertIsNotNone(model.refinement_readout.weight.grad)
        self.assertGreater(float(model.refinement_readout.weight.grad.abs().sum()), 0.0)

    def test_empty_sequence_and_no_grad_evaluation(self):
        model = fixed_model().eval()
        with torch.no_grad():
            self.assertEqual(model([], self.relations)["predictions"].numel(), 0)
            output = model(self.sequence.events, self.relations)
        self.assertFalse(output["predictions"].requires_grad)
        self.assertTrue(bool(torch.isfinite(output["predictions"]).all()))


class ObjectiveChecks(unittest.TestCase):
    @staticmethod
    def output():
        return dict(long_logits=torch.tensor([8.0, -1.0, 0.5, 1.0], requires_grad=True),
                    short_logits=torch.tensor([-7.0, 0.2, -0.3, 0.4], requires_grad=True),
                    labels=torch.tensor([0.0, 0.0, 1.0, 1.0]),
                    valid_mask=torch.tensor([False, True, True, True]),
                    triggered=torch.tensor([False, False, False, False]),
                    base_weights=[torch.ones(1) for _ in range(4)],
                    refined_weights=[None] * 4,
                    coefficients=[torch.tensor([0.2, 0.3])] * 4,
                    distinguishability=[torch.tensor([0.4, 0.7])] * 4)

    def test_losses_match_independent_numpy_calculation(self):
        output = self.output()
        weights = LossWeights(l2=0)
        losses = compute_objective(output, weights=weights)
        long = np.array([-1.0, 0.5, 1.0])
        short = np.array([0.2, -0.3, 0.4])
        full = long + short
        truth = np.array([0, 1, 1])
        prediction = np.mean(np.logaddexp(0, full) - truth * full)
        a, b = long - long.mean(), short - short.mean()
        correlation = (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)) ** 2
        structure = np.sum((1 - np.array([0.4, 0.7])) * np.array([0.2, 0.3]) ** 2)
        self.assertAlmostEqual(float(losses["prediction"].detach()), float(prediction), places=6)
        self.assertAlmostEqual(float(losses["correlation"].detach()), float(correlation), places=6)
        self.assertAlmostEqual(float(losses["structure"].detach()), float(structure), places=6)
        self.assertEqual(float(losses["valid_events"]), 3)

    def test_refined_consistency_target_has_no_gradient(self):
        base = torch.tensor([0.7, 0.3], requires_grad=True)
        target = torch.tensor([0.2, 0.8], requires_grad=True)
        jensen_shannon(base, target).backward()
        self.assertIsNotNone(base.grad)
        self.assertIsNone(target.grad)

    def test_no_refinement_has_zero_consistency(self):
        loss = compute_objective(self.output(), weights=LossWeights(l2=0))
        self.assertEqual(float(loss["consistency"].detach()), 0.0)

    def test_masked_first_event_is_excluded_from_all_statistics(self):
        original = self.output()
        changed = self.output()
        with torch.no_grad():
            changed["long_logits"][0] = -100
            changed["short_logits"][0] = 100
        first = compute_objective(original, weights=LossWeights(l2=0))
        second = compute_objective(changed, weights=LossWeights(l2=0))
        for key in ("prediction", "separation", "structure", "consistency"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_effective_batch_centering_is_event_weighted(self):
        output = self.output()
        split = []
        for start, end in ((0, 2), (2, 4)):
            split.append({key: value[start:end] for key, value in output.items()})
        joined = compute_objective(output, weights=LossWeights(l2=0))
        partitioned = compute_objective(split, weights=LossWeights(l2=0))
        for key in ("total", "correlation", "prediction", "structure"):
            torch.testing.assert_close(joined[key], partitioned[key])


class MetricChecks(unittest.TestCase):
    def test_auc_matches_independent_pairwise_comparisons(self):
        rng = np.random.default_rng(41)
        labels = np.array([0, 1] * 25)
        scores = np.round(rng.uniform(size=50), 1)
        differences = scores[labels == 1, None] - scores[labels == 0][None, :]
        expected = ((differences > 0).astype(float) + 0.5 * (differences == 0)).mean()
        self.assertAlmostEqual(binary_metrics(labels, scores)["auc"], float(expected), places=12)

    def test_auc_ties_and_single_class(self):
        self.assertEqual(binary_metrics([0, 1], [0.5, 0.5])["auc"], 0.5)
        self.assertIsNone(binary_metrics([1, 1], [0.1, 0.9])["auc"])
        self.assertIsNone(binary_metrics([], [])["auc"])

    def test_seed_summary_uses_sample_standard_deviation(self):
        summary = summarize_seeds([dict(auc=0.6), dict(auc=0.8)])
        self.assertAlmostEqual(summary["metrics"]["auc"]["mean"], 0.7)
        self.assertAlmostEqual(summary["metrics"]["auc"]["std"], math.sqrt(0.02))

    def test_atomic_groups_and_no_duplicate_windows(self):
        events = list(learner(length=7).events)
        events[1] = replace(events[1], group_id="g")
        events[2] = replace(events[2], group_id="g")
        chunks = list(iter_event_chunks(events, 2, 2))
        self.assertEqual([event.event_id for chunk in chunks for event in chunk], [event.event_id for event in events])
        self.assertTrue(any([event.group_id for event in chunk].count("g") == 2 for chunk in chunks))

    def test_online_event_count(self):
        sequences = [learner(f"s{i}") for i in range(3)]
        metrics = evaluate_model(fixed_model(), sequences, build_cooccurrence(sequences, 3), 8, 4)
        self.assertEqual(metrics["events"], sum(len(sequence.events) - 1 for sequence in sequences))


class ExperimentChecks(unittest.TestCase):
    def test_training_checkpoint_and_held_out_reload(self):
        from train import evaluate_checkpoint, train_one_seed
        sequences = [learner(f"s{i}", 8, i) for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_dataset(sequences, {"num_items": 4, "num_concepts": 3,
                                       "items": ["0", "1", "2", "3"],
                                       "concepts": ["0", "1", "2"]}, root / "data",
                            "integrity", counts=(4, 1, 1))
            settings = TrainingConfig(epochs=2, batch_size=2, first_window=8, stride=4, patience=1)
            with __import__("contextlib").redirect_stdout(io.StringIO()):
                result = train_one_seed(root / "data" / "canonical.csv", "integrity", root / "runs",
                                        small_config(), settings, torch.device("cpu"))
            restored = evaluate_checkpoint(Path(result["checkpoint"]), root / "data" / "canonical.csv", "test", "cpu")
            self.assertEqual(result["test"], restored["metrics"])
            self.assertIn(result["selected_epoch"], (1, 2))
            path = root / "data" / "canonical.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_checkpoint(Path(result["checkpoint"]), path, "test", "cpu")

    def test_bfloat16_checkpoint_preserves_assessment_precision(self):
        from train import evaluate_checkpoint, train_one_seed
        sequences = [learner(f"s{i}", 6, i) for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocabulary = {"num_items": 4, "num_concepts": 3,
                          "items": ["0", "1", "2", "3"], "concepts": ["0", "1", "2"]}
            prepare_dataset(sequences, vocabulary, root / "data", "integrity", counts=(4, 1, 1))
            settings = TrainingConfig(epochs=1, batch_size=2, first_window=8, stride=4, precision="bfloat16")
            with __import__("contextlib").redirect_stdout(io.StringIO()):
                result = train_one_seed(root / "data" / "canonical.csv", "integrity", root / "runs",
                                        small_config(), settings, torch.device("cpu"))
            restored = evaluate_checkpoint(Path(result["checkpoint"]), root / "data" / "canonical.csv", "test", "cpu")
            self.assertEqual(result["test"], restored["metrics"])


class IntegrityResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "failed"})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "error"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IASE-KT software integrity.")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=IntegrityResult)
    result = runner.run(suite)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "scope": "Mathematical, temporal, data, and execution integrity",
            "public_dataset_training": False,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "device": "cpu", "tests": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "passed": result.wasSuccessful(), "checks": result.records,
        }
        arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
