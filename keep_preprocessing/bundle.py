"""Bundle serialization and validation; uses only the Python standard library."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from . import SCHEMA_VERSION
from .projection import project, timestamp

TABLES = {
    "concepts.tsv": ["row_id", "concept_id", "label", "root_depth"],
    "edges.tsv": ["parent_id", "child_id"],
    "rollup.tsv": ["source_id", "target_id"],
    "rollup_status.tsv": ["source_id", "root_depth", "reason"],
    "counts.tsv": ["row", "column", "count"],
}
PAYLOADS = (*TABLES, "histories.jsonl", "prediction_histories.jsonl")


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if line != canonical_json(row) + "\n":
                raise ValueError("Noncanonical JSONL serialization")
            yield row


def write_table(directory, name, rows):
    with (Path(directory) / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(TABLES[name])
        writer.writerows(rows)


def read_table(directory, name):
    with (Path(directory) / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != TABLES[name]:
            raise ValueError(f"Invalid columns: {name}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Malformed rows: {name}")
        canonical = io.StringIO(newline="")
        writer = csv.writer(canonical, delimiter="\t", lineterminator="\n")
        writer.writerow(TABLES[name])
        writer.writerows([row[column] for column in TABLES[name]] for row in rows)
        if (Path(directory) / name).read_bytes() != canonical.getvalue().encode():
            raise ValueError(f"Noncanonical table serialization: {name}")
        return rows


def count_histories(rows, concept_ids, fit_patients=None):
    """Independent patient-set oracle, used to check the library's sparse output."""
    axes = {code: i for i, code in enumerate(concept_ids)}
    counts = Counter()
    occurrences = Counter()
    current = None
    seen = set()
    patients = set()
    present_patients = 0

    def flush(patient):
        nonlocal present_patients
        if patient is None or (fit_patients is not None and patient not in fit_patients):
            return
        present = sorted(axes[code] for code, n in occurrences.items() if n >= 2)
        present_patients += bool(present)
        counts.update((i, j) for i in present for j in present)

    for row in rows:
        patient = row["patient_id"]
        if patient != current:
            flush(current)
            if patient in patients:
                raise ValueError("Histories must be grouped by patient")
            patients.add(patient)
            current = patient
            occurrences.clear()
            seen.clear()
        if row["diagnosis_id"] in seen:
            raise ValueError("Duplicate diagnosis in exported histories")
        seen.add(row["diagnosis_id"])
        targets = row["rolled_up_concepts"]
        if targets != sorted(set(targets), key=int) or not set(targets) <= axes.keys():
            raise ValueError("Invalid diagnosis target set")
        occurrences.update(targets)
    flush(current)
    return dict(sorted(counts.items())), present_patients


def validate(directory):
    """Verify hashes, canonical index, graph, histories, projection and exact counts.

    This is an internal-consistency check. Fixture and author comparisons are
    separate evidence, never implied by a valid bundle.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest_path.read_text() != canonical_json(manifest) + "\n":
        raise ValueError("Noncanonical manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported bundle schema")
    if set(manifest["artifacts"]) != set(PAYLOADS):
        raise ValueError("Incomplete artifact manifest")
    for name in PAYLOADS:
        if digest(directory / name) != manifest["artifacts"][name]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    concepts = read_table(directory, "concepts.tsv")
    ids = [row["concept_id"] for row in concepts]
    if ids != sorted(set(ids), key=int) or any(
        int(code) <= 0 or str(int(code)) != code for code in ids
    ):
        raise ValueError("Concept IDs must be unique, canonical and numerically sorted")
    if [row["row_id"] for row in concepts] != list(map(str, range(len(ids)))):
        raise ValueError("Invalid contiguous concept index")
    if manifest["shape"] != [len(ids), len(ids)]:
        raise ValueError("Matrix shape does not match the concept index")
    parameters = manifest["parameters"]
    fixed_parameters = {
        "root": "4274025",
        "depth_measure": "min",
        "include_root": False,
        "orphan_rescue": False,
        "presence_threshold": 2,
        "threshold_stage": "post_rollup",
        "diagonal": "patient_prevalence",
        "history_window": "complete",
        "count_unit": "patient",
        "mimic4_time": "admission",
    }
    if any(parameters.get(key) != value for key, value in fixed_parameters.items()):
        raise ValueError("Unsupported preprocessing semantics")
    if parameters.get("target_policy") not in {"all_retained", "first_retained"} or parameters.get(
        "path_source"
    ) not in {"direct", "closure"}:
        raise ValueError("Unknown roll-up configuration")
    if type(parameters.get("max_depth")) is not int or parameters["max_depth"] < 0:
        raise ValueError("Invalid depth cutoff")
    if parameters["include_root"] or parameters["orphan_rescue"]:
        raise ValueError("Schema v1 baseline excludes root and orphan rescue")
    if any(not 1 <= int(row["root_depth"]) <= parameters["max_depth"] for row in concepts):
        raise ValueError("Invalid retained root depth")
    id_set = set(ids)
    edges = [(row["parent_id"], row["child_id"]) for row in read_table(directory, "edges.tsv")]
    if edges != sorted(set(edges), key=lambda e: tuple(map(int, e))):
        raise ValueError("Edges must be canonical and unique")
    if any(p not in id_set or c not in id_set or p == c for p, c in edges):
        raise ValueError("Invalid graph endpoint")
    incoming = Counter(c for _, c in edges)
    children = {}
    for parent, child in edges:
        children.setdefault(parent, []).append(child)
    pending = [code for code in ids if not incoming[code]]
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        for child in children.get(node, []):
            incoming[child] -= 1
            if incoming[child] == 0:
                pending.append(child)
    if visited != len(ids):
        raise ValueError("Exported graph contains a cycle")
    if manifest["graph"] != {
        "direction": "parent_to_child",
        "directed": True,
        "include_isolates": True,
    }:
        raise ValueError("Unsupported effective topology")
    targets = {}
    pairs = [(r["source_id"], r["target_id"]) for r in read_table(directory, "rollup.tsv")]
    if pairs != sorted(set(pairs), key=lambda e: tuple(map(int, e))):
        raise ValueError("Roll-up pairs must be canonical and unique")
    for source, target in pairs:
        if target not in id_set:
            raise ValueError("Roll-up target outside shared index")
        targets.setdefault(source, set()).add(target)
    for code in ids:
        if targets.get(code) != {code}:
            raise ValueError("Retained concepts must map to themselves")
    status_rows = read_table(directory, "rollup_status.tsv")
    status = {r["source_id"]: r for r in status_rows}
    if list(status) != sorted(set(status), key=int) or len(status) != len(status_rows):
        raise ValueError("Noncanonical roll-up status index")
    if not set(targets) <= status.keys():
        raise ValueError("Missing roll-up status")
    for code, row in status.items():
        if (row["reason"] in {"retained", "rolled_up"}) != bool(targets.get(code)):
            raise ValueError("Roll-up status disagrees with target presence")
    previous = None
    diagnoses = 0
    losses = Counter()
    patient_ids = set(manifest["patient_ids"])
    if manifest["patient_ids"] != sorted(patient_ids):
        raise ValueError("Noncanonical patient index")
    for row in read_jsonl(directory / "histories.jsonl"):
        diagnoses += 1
        key = (row["patient_id"], row["diagnosis_id"])
        if previous is not None and key <= previous:
            raise ValueError("Histories must have unique canonical order")
        previous = key
        if row["patient_id"] not in patient_ids or not row["diagnosis_id"]:
            raise ValueError("Invalid history identity")
        if row["event_time"] != timestamp(row["event_time"]):
            raise ValueError("Noncanonical event time")
        mapped = row["mapped_concepts"]
        if mapped != sorted(set(mapped), key=int):
            raise ValueError("Invalid mapped concept order")
        expected_targets = set().union(*(targets.get(code, set()) for code in mapped))
        if expected_targets != set(row["rolled_up_concepts"]):
            raise ValueError("History disagrees with exported roll-up")
        outcome = row["mapping_outcome"]
        if outcome != "mapped":
            if outcome not in {
                "missing_source",
                "unsupported_source_format",
                "empty_icd_mapping",
                "empty_rollup",
            }:
                raise ValueError("Unknown mapping outcome")
            if expected_targets or (outcome != "empty_rollup" and mapped):
                raise ValueError("Mapping failure disagrees with concepts")
            losses[outcome] += 1
        elif not expected_targets:
            raise ValueError("Mapped diagnosis has no targets")
    fit = manifest["fit_patient_ids"]
    if fit != sorted(set(fit)) or not set(fit) <= patient_ids:
        raise ValueError("Invalid fitting cohort")
    expected_counts, present = count_histories(
        read_jsonl(directory / "histories.jsonl"), ids, set(fit)
    )
    triples = [
        (int(r["row"]), int(r["column"]), int(r["count"]))
        for r in read_table(directory, "counts.tsv")
    ]
    if triples != [(i, j, n) for (i, j), n in expected_counts.items()]:
        raise ValueError("Raw counts disagree with patient-level histories")
    if manifest["statistics"] != {
        "patients": len(patient_ids),
        "fit_patients": len(fit),
        "diagnoses": diagnoses,
        "patients_with_present_targets": present,
        "mapping_losses": dict(sorted(losses.items())),
    }:
        raise ValueError("Manifest statistics disagree with histories")
    cohort = {
        p: (value["label"], value["index_date"])
        for p, value in manifest["projection_cohort"].items()
    }
    if manifest["projection_policy"] != {
        "positive": "strictly_before_index_date",
        "control": "uncensored",
        "ordering": "second_occurrence_then_numeric_concept_id",
        "threshold": 2,
    }:
        raise ValueError("Unsupported prediction-history policy")
    if any(
        label not in {0, 1}
        or (label == 1 and date is None)
        or (label == 0 and date is not None)
        or date != timestamp(date)
        for label, date in cohort.values()
    ):
        raise ValueError("Invalid projection cohort")
    if not set(cohort) <= patient_ids:
        raise ValueError("Projection cohort contains unknown patients")
    projection = list(read_jsonl(directory / "prediction_histories.jsonl"))
    if projection != project(read_jsonl(directory / "histories.jsonl"), cohort):
        raise ValueError("Prediction history disagrees with observation policy")
    return {
        "valid": True,
        "shape": manifest["shape"],
        "nonzero_counts": len(triples),
        "artifact_sha256": manifest["artifacts"],
    }
