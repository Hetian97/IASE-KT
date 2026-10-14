"""Complete-response ingestion, student splits, and training-only KC relations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class Event:
    student_id: str
    item_id: int
    concept_ids: tuple[int, ...]
    outcome: int
    position: int
    event_id: str
    group_id: str | None = None
    original_row: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.student_id, str) or not self.student_id or self.student_id != self.student_id.strip():
            raise ValueError("student_id must be nonempty and stripped")
        if not isinstance(self.event_id, str) or not self.event_id or self.event_id != self.event_id.strip():
            raise ValueError("event_id must be nonempty")
        if isinstance(self.item_id, bool) or not isinstance(self.item_id, int) or self.item_id < 0:
            raise ValueError("item_id must be a nonnegative integer")
        if not isinstance(self.concept_ids, tuple) or not self.concept_ids:
            raise ValueError("concept_ids must be a nonempty tuple")
        if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in self.concept_ids):
            raise ValueError("concept IDs must be nonnegative integers")
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("concept IDs must be unique within an event")
        if isinstance(self.outcome, bool) or not isinstance(self.outcome, int) or self.outcome not in (0, 1):
            raise ValueError("outcome must be integer 0 or 1")
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 0:
            raise ValueError("position must be a nonnegative integer")
        if self.group_id is not None and (not isinstance(self.group_id, str) or not self.group_id or self.group_id != self.group_id.strip()):
            raise ValueError("group_id must be None or a nonempty stripped string")
        if self.original_row is not None and (isinstance(self.original_row, bool) or not isinstance(self.original_row, int) or self.original_row < 0):
            raise ValueError("original_row must be None or a nonnegative integer")


@dataclass(frozen=True)
class StudentSequence:
    student_id: str
    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.student_id, str) or not self.student_id or self.student_id != self.student_id.strip():
            raise ValueError("student_id must be nonempty and stripped")
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("events must be a nonempty tuple")
        if any(event.student_id != self.student_id for event in self.events):
            raise ValueError("Each event must belong to its StudentSequence")
        if any(a.position > b.position for a, b in zip(self.events, self.events[1:])):
            raise ValueError("Events must be ordered by position")
        if any(a.position == b.position and a.original_row is not None
               and b.original_row is not None and a.original_row > b.original_row
               for a, b in zip(self.events, self.events[1:])):
            raise ValueError("Equal-position events must follow their original row order")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event_id must be unique within a student")
        closed_groups: set[str] = set()
        previous_group: str | None = None
        for event in self.events:
            if event.group_id != previous_group:
                if previous_group is not None:
                    closed_groups.add(previous_group)
                if event.group_id is not None and event.group_id in closed_groups:
                    raise ValueError("Each group_id must occupy one contiguous block per student")
                previous_group = event.group_id


CSV_COLUMNS = ("student_id", "item_id", "concept_ids", "outcome", "position", "event_id", "group_id", "original_row")


def _merge_events(events: Iterable[Event]) -> list[StudentSequence]:
    grouped: dict[str, dict[str, tuple[int, Event]]] = defaultdict(dict)
    for row_index, event in enumerate(events):
        previous = grouped[event.student_id].get(event.event_id)
        if previous is None:
            grouped[event.student_id][event.event_id] = (row_index, event)
            continue
        first_row, old = previous
        if (old.item_id, old.outcome, old.position, old.group_id, old.original_row) != (
            event.item_id, event.outcome, event.position, event.group_id, event.original_row
        ):
            raise ValueError(f"Conflicting metadata for event {event.student_id}/{event.event_id}")
        merged = tuple(sorted(set(old.concept_ids) | set(event.concept_ids)))
        grouped[event.student_id][event.event_id] = (
            first_row, Event(old.student_id, old.item_id, merged, old.outcome,
                             old.position, old.event_id, old.group_id, old.original_row)
        )
    return [StudentSequence(student, tuple(event for _, event in sorted(
        records.values(), key=lambda pair: (pair[1].position,
                                          pair[1].original_row if pair[1].original_row is not None else pair[0])
    ))) for student, records in sorted(grouped.items())]


def load_sequences(path: str | Path) -> list[StudentSequence]:
    """Read canonical CSV and merge only duplicate event IDs with consistent metadata."""
    events: list[Event] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_COLUMNS[:6]) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing canonical CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                events.append(Event(
                    row["student_id"].strip(), int(row["item_id"]),
                    tuple(sorted({int(k.strip()) for k in row["concept_ids"].split("|") if k.strip()})),
                    int(row["outcome"]), int(row["position"]), row["event_id"].strip(),
                    (row.get("group_id") or "").strip() or None,
                    int(row["original_row"]) if (row.get("original_row") or "").strip() else None,
                ))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid canonical CSV row {line}: {exc}") from exc
    if not events:
        raise ValueError("The canonical CSV has no valid events")
    return _merge_events(events)


def write_sequences(sequences: Sequence[StudentSequence], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for sequence in sequences:
            for event in sequence.events:
                writer.writerow({
                    "student_id": event.student_id, "item_id": event.item_id,
                    "concept_ids": "|".join(map(str, event.concept_ids)), "outcome": event.outcome,
                    "position": event.position, "event_id": event.event_id,
                    "group_id": event.group_id or "",
                    "original_row": event.original_row if event.original_row is not None else "",
                })


def _hash_student(student: str, dataset: str, purpose: str) -> bytes:
    return hashlib.sha256(f"IASEKT-v1|{purpose}|{dataset}|{student.strip()}".encode("utf-8")).digest()


def split_students(
    sequences: Sequence[StudentSequence], dataset: str,
    counts: tuple[int, int, int] | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2), sample_size: int | None = None,
) -> dict[str, list[StudentSequence]]:
    """Apply the manuscript's deterministic sampling and disjoint student partition."""
    if not isinstance(dataset, str) or not dataset or dataset != dataset.strip():
        raise ValueError("dataset must be a nonempty stripped name")
    if len({sequence.student_id for sequence in sequences}) != len(sequences):
        raise ValueError("Input sequences contain duplicate student IDs")
    students = list(sequences)
    if sample_size is not None:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 1 <= sample_size <= len(students):
            raise ValueError("sample_size must lie between 1 and the number of students")
        students = sorted(students, key=lambda seq: (_hash_student(seq.student_id, dataset, "sample"), seq.student_id))[:sample_size]
    students.sort(key=lambda seq: (_hash_student(seq.student_id, dataset, "split"), seq.student_id))
    n = len(students)
    if counts is not None:
        if len(counts) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
            raise ValueError("counts must contain three nonnegative integers")
        if sum(counts) != n:
            raise ValueError(f"Split counts total {sum(counts)}, but {n} students are available")
        if n >= 3 and any(v == 0 for v in counts):
            raise ValueError("All three partitions must be nonempty when at least three students exist")
        allocation = list(counts)
    else:
        if len(ratios) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not math.isfinite(v) or v <= 0 for v in ratios):
            raise ValueError("ratios must contain three finite positive values")
        total = sum(ratios)
        targets = [n * ratio / total for ratio in ratios]
        allocation = [math.floor(value) for value in targets]
        residual_order = sorted(range(3), key=lambda i: (-(targets[i] - allocation[i]), i))
        for index in residual_order[:n - sum(allocation)]:
            allocation[index] += 1
        if n >= 3:
            for index in range(3):
                if allocation[index] == 0:
                    donor = max((i for i in range(3) if allocation[i] > 1),
                                key=lambda i: (allocation[i] - targets[i], allocation[i], -i))
                    allocation[donor] -= 1
                    allocation[index] += 1
    train_end = allocation[0]
    validation_end = train_end + allocation[1]
    return {"train": students[:train_end], "validation": students[train_end:validation_end],
            "test": students[validation_end:]}


def read_split(sequences: Sequence[StudentSequence], path: str | Path) -> dict[str, list[StudentSequence]]:
    """Load a saved partition and require exactly one assignment per student."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A split file must contain a JSON object")
    partitions = payload.get("students", payload)
    names = ("train", "validation", "test")
    if not isinstance(partitions, dict) or any(name not in partitions for name in names):
        raise ValueError("A split must contain train, validation, and test")
    if any(not isinstance(partitions[name], list)
           or any(not isinstance(student, str) or not student or student != student.strip()
                  for student in partitions[name]) for name in names):
        raise ValueError("Each split must contain a list of nonempty stripped student IDs")
    ids = [student for name in names for student in partitions[name]]
    by_student = {sequence.student_id: sequence for sequence in sequences}
    if len(by_student) != len(sequences) or len(ids) != len(set(ids)):
        raise ValueError("Student assignments must be unique")
    if set(ids) != set(by_student):
        raise ValueError("Saved student assignments do not match the canonical CSV")
    return {name: [by_student[student] for student in partitions[name]] for name in names}


class ConceptRelations:
    """Sparse Jaccard overlaps measured on distinct training items."""

    def __init__(self, num_concepts: int, item_counts: Sequence[int],
                 intersections: dict[tuple[int, int], int], epsilon: float = 1e-8) -> None:
        if isinstance(num_concepts, bool) or not isinstance(num_concepts, int) or num_concepts <= 0:
            raise ValueError("num_concepts must be a positive integer")
        if len(item_counts) != num_concepts or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in item_counts):
            raise ValueError("item_counts must contain one nonnegative integer per concept")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.num_concepts = num_concepts
        self.item_counts = tuple(item_counts)
        self.intersections = dict(intersections)
        self.epsilon = epsilon
        for (first, second), count in self.intersections.items():
            if not 0 <= first < second < num_concepts or isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= min(self.item_counts[first], self.item_counts[second]):
                raise ValueError("Invalid sparse concept intersection")

    def overlap(self, first: int, second: int) -> float:
        if not 0 <= first < self.num_concepts or not 0 <= second < self.num_concepts:
            raise IndexError("Concept ID lies outside the fitted vocabulary")
        if first == second:
            count = self.item_counts[first]
            return count / (count + self.epsilon)
        shared = self.intersections.get((min(first, second), max(first, second)), 0)
        union = self.item_counts[first] + self.item_counts[second] - shared
        return shared / (union + self.epsilon)

    def distinguishability(self, concept_ids: Sequence[int], device: Any = None,
                          dtype: torch.dtype | None = None) -> torch.Tensor:
        ids = tuple(int(concept) for concept in concept_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("concept_ids must be nonempty and unique")
        if any(k < 0 or k >= self.num_concepts for k in ids):
            raise IndexError("Concept ID lies outside the fitted vocabulary")
        values = [1.0 - max((self.overlap(k, j) for j in ids if j != k), default=0.0) for k in ids]
        return torch.tensor(values, dtype=dtype or torch.float32, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {"num_concepts": self.num_concepts, "item_counts": list(self.item_counts),
                "intersections": [[a, b, count] for (a, b), count in sorted(self.intersections.items())],
                "epsilon": self.epsilon}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ConceptRelations:
        rows = values["intersections"]
        if len({(a, b) for a, b, _ in rows}) != len(rows):
            raise ValueError("Serialized intersections contain duplicate concept pairs")
        return cls(values["num_concepts"], values["item_counts"],
                   {(a, b): count for a, b, count in rows},
                   values.get("epsilon", 1e-8))


def build_cooccurrence(train_sequences: Sequence[StudentSequence], num_concepts: int) -> ConceptRelations:
    """Fit relationships from unique training items, never response frequencies."""
    if isinstance(num_concepts, bool) or not isinstance(num_concepts, int) or num_concepts <= 0:
        raise ValueError("num_concepts must be a positive integer")
    item_concepts: dict[int, tuple[int, ...]] = {}
    for sequence in train_sequences:
        for event in sequence.events:
            concepts = tuple(sorted(event.concept_ids))
            old = item_concepts.setdefault(event.item_id, concepts)
            if old != concepts:
                raise ValueError(f"Training item {event.item_id} has inconsistent complete KC annotation")
    counts = [0] * num_concepts
    shared: Counter[tuple[int, int]] = Counter()
    for concepts in item_concepts.values():
        for concept in concepts:
            if concept < 0 or concept >= num_concepts:
                raise ValueError("A training concept lies outside the vocabulary")
            counts[concept] += 1
        shared.update(itertools.combinations(concepts, 2))
    return ConceptRelations(num_concepts, counts, dict(shared))


def _natural_key(value: str) -> tuple[Any, ...]:
    parts = tuple((0, int(part)) if part.isdigit() else (1, part)
                  for part in re.split(r"(\d+)", value))
    return parts, value


def _encode_records(records: Sequence[dict[str, Any]]) -> tuple[list[StudentSequence], dict[str, Any]]:
    if not records:
        raise ValueError("No valid complete responses were retained")
    items = sorted({row["item"] for row in records}, key=_natural_key)
    concepts = sorted({concept for row in records for concept in row["concepts"]}, key=_natural_key)
    item_ids, concept_ids = {item: i for i, item in enumerate(items)}, {kc: i for i, kc in enumerate(concepts)}
    events = [Event(row["student"], item_ids[row["item"]],
                    tuple(sorted({concept_ids[k] for k in row["concepts"]})),
                    row["outcome"], row["position"], row["event"], row.get("group"),
                    row.get("original_row")) for row in records]
    return _merge_events(events), {"num_items": len(items), "num_concepts": len(concepts),
                                    "items": items, "concepts": concepts}


def import_ednet(kt1_directory: str | Path, questions_path: str | Path,
                 sample_size: int | None = None, dataset: str = "EdNet-KT1-50K"
                 ) -> tuple[list[StudentSequence], dict[str, Any]]:
    """Read official KT1 student CSVs and questions.csv tags and answer keys."""
    metadata: dict[str, tuple[tuple[str, ...], str]] = {}
    with Path(questions_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question_id", "correct_answer", "tags"}
        if required - set(reader.fieldnames or ()):
            raise ValueError("EdNet questions.csv must contain question_id, correct_answer, and tags")
        for row in reader:
            question = row["question_id"].strip()
            tag_values = [tag.strip() for tag in row["tags"].split(";") if tag.strip()]
            if any(not tag.isdigit() for tag in tag_values):
                continue
            tags = tuple(sorted(set(tag_values), key=_natural_key))
            answer = row["correct_answer"].strip()
            if question and tags and answer in {"a", "b", "c", "d"}:
                entry = (tags, answer)
                if question in metadata and metadata[question] != entry:
                    raise ValueError(f"Conflicting metadata for EdNet question {question}")
                metadata[question] = entry
    files = sorted(Path(kt1_directory).glob("*.csv"), key=lambda path: _natural_key(path.name))
    if sample_size is not None:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 1 <= sample_size <= len(files):
            raise ValueError("sample_size must lie between 1 and the number of EdNet student files")
        files = sorted(files, key=lambda path: (_hash_student(path.stem.strip(), dataset, "sample"), path.stem))[:sample_size]
    items = sorted(metadata, key=_natural_key)
    concepts = sorted({tag for tags, _ in metadata.values() for tag in tags}, key=_natural_key)
    if not items or not concepts:
        raise ValueError("EdNet question metadata has no valid item and KC annotations")
    item_ids = {item: i for i, item in enumerate(items)}
    concept_ids = {concept: i for i, concept in enumerate(concepts)}
    sequences: list[StudentSequence] = []
    for path in files:
        student = path.stem.strip()
        events: list[Event] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"timestamp", "solving_id", "question_id", "user_answer"}
            if required - set(reader.fieldnames or ()):
                raise ValueError(f"EdNet KT1 columns are missing in {path.name}")
            for index, row in enumerate(reader):
                question = row["question_id"].strip()
                answer = row["user_answer"].strip()
                if question not in metadata or answer not in {"a", "b", "c", "d"}:
                    continue
                tags, correct = metadata[question]
                events.append(Event(student, item_ids[question], tuple(sorted(concept_ids[tag] for tag in tags)),
                                    int(answer == correct), int(row["timestamp"]), f"{student}:{index}",
                                    row["solving_id"].strip() or None, index))
        if events:
            events.sort(key=lambda event: event.position)
            sequences.append(StudentSequence(student, tuple(events)))
        elif sample_size is not None:
            raise ValueError(f"Selected EdNet student {student} has no valid complete responses")
    if not sequences:
        raise ValueError("No valid complete EdNet responses were retained")
    sequences.sort(key=lambda sequence: sequence.student_id)
    vocabulary = {"num_items": len(items), "num_concepts": len(concepts),
                  "items": items, "concepts": concepts, "source_format": "EdNet-KT1"}
    return sequences, vocabulary


def import_assist(path: str | Path) -> tuple[list[StudentSequence], dict[str, Any]]:
    """Read ASSIST2009 skill-builder rows and merge skills by student and order_id."""
    records: list[dict[str, Any]] = []
    first_rows: dict[tuple[str, int], int] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "problem_id", "skill_id", "correct", "order_id"}
        if required - set(reader.fieldnames or ()):
            raise ValueError("ASSIST CSV must contain user_id, problem_id, skill_id, correct, and order_id")
        for index, row in enumerate(reader):
            student, item = row["user_id"].strip(), row["problem_id"].strip()
            skills = tuple(sorted({kc.strip() for kc in re.split(r"[;|]", row["skill_id"]) if kc.strip() and kc.strip().lower() not in {"nan", "none", "-1"}}, key=_natural_key))
            correct = row["correct"].strip()
            if not student or not item or item.lower() in {"nan", "none", "-1"} or not skills or correct not in {"0", "1", "0.0", "1.0"}:
                continue
            try:
                order = int(row["order_id"])
            except ValueError:
                continue
            if order < 0:
                continue
            first_row = first_rows.setdefault((student, order), index)
            records.append({"student": student, "item": item, "concepts": skills,
                            "outcome": int(float(correct)), "position": order,
                            "event": f"{student}:{order}", "original_row": first_row})
    sequences, vocabulary = _encode_records(records)
    vocabulary["source_format"] = "ASSIST2009-skill-builder"
    return sequences, vocabulary


def prepare_dataset(sequences: Sequence[StudentSequence], vocabulary: dict[str, Any],
                    output: str | Path, dataset: str, counts: tuple[int, int, int] | None = None,
                    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
                    sample_size: int | None = None) -> dict[str, list[StudentSequence]]:
    splits = split_students(sequences, dataset, counts, ratios, sample_size)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    selected = [sequence for partition in splits.values() for sequence in partition]
    write_sequences(selected, directory / "canonical.csv")
    payload = {"dataset": dataset, "hash_protocol": "IASEKT-v1",
               "students": {name: [sequence.student_id for sequence in rows] for name, rows in splits.items()}}
    (directory / "split.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (directory / "vocab.json").write_text(json.dumps(vocabulary, indent=2) + "\n", encoding="utf-8")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare complete-response datasets for IASE-KT.")
    parser.add_argument("format", choices=("canonical", "ednet", "assist"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--questions", type=Path, help="Official EdNet questions.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--counts", type=int, nargs=3)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    args = parser.parse_args()
    if args.format == "ednet":
        if args.questions is None:
            parser.error("--questions is required for EdNet")
        sequences, vocabulary = import_ednet(args.input, args.questions, args.sample_size, args.dataset)
    elif args.format == "assist":
        sequences, vocabulary = import_assist(args.input)
    else:
        source_sequences = load_sequences(args.input)
        records = [{"student": event.student_id, "item": str(event.item_id),
                    "concepts": tuple(map(str, event.concept_ids)), "outcome": event.outcome,
                    "position": event.position, "event": event.event_id, "group": event.group_id,
                    "original_row": event.original_row}
                   for sequence in source_sequences for event in sequence.events]
        sequences, vocabulary = _encode_records(records)
        vocabulary["source_format"] = "canonical"
    splits = prepare_dataset(sequences, vocabulary, args.output, args.dataset,
                             tuple(args.counts) if args.counts else None,
                             tuple(args.ratios), args.sample_size)
    print(json.dumps({name: {"students": len(rows), "events": sum(len(seq.events) for seq in rows)}
                      for name, rows in splits.items()}, indent=2))


if __name__ == "__main__":
    main()
