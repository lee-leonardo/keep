"""Versioned comparison interchange, with public loading and integrity checks.

Validation establishes internal consistency, never cross-implementation parity.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from . import SCHEMA_VERSION
from .config import MAPPING_CONTRACT, TIMESTAMP_CONTRACT, checked_settings
from .counts import count_cooccurrence
from .io import (
    concept_id,
    event_time,
    file_digest,
    opaque_id,
    read_jsonl,
    read_tsv,
    write_json,
    write_jsonl,
    write_tsv,
)
from .ontology import require_dag

TABLES = {
    "concepts.tsv": ["row_id", "concept_id", "label", "root_depth"],
    "edges.tsv": ["parent_id", "child_id", "relation"],
    "uncut_concepts.tsv": ["row_id", "concept_id", "label", "root_depth"],
    "uncut_edges.tsv": ["parent_id", "child_id", "relation"],
    "closure_links.tsv": ["ancestor_id", "descendant_id", "distance"],
    "rollup.tsv": ["source_id", "target_id"],
    "rollup_status.tsv": ["source_id", "root_depth", "reason"],
    "counts.tsv": ["row", "column", "count"],
}
JSONL_FILES = ("histories.jsonl", "rolled_histories.jsonl", "patient_presence.jsonl")
PAYLOADS = (*TABLES, *JSONL_FILES)
RAW_FIELDS = {
    "patient_id",
    "diagnosis_id",
    "visit_id",
    "event_time",
    "source_value",
    "source_concept_id",
    "mapped_concepts",
    "unmapped_reason",
}
BOUNDARIES = ("histories", "ontology", "rollup", "counts")


@dataclass
class BoundaryBundle:
    manifest: dict
    tables: dict
    histories: list
    rolled_histories: list
    patient_presence: list

    def counts_by_concept(self):
        """Decode both native axes independently; absent coordinates are zero."""
        rows, columns = self.manifest["row_concept_ids"], self.manifest["column_concept_ids"]
        return {
            (rows[int(r["row"])], columns[int(r["column"])]): int(r["count"])
            for r in self.tables["counts.tsv"]
        }


def load_bundle(directory, *, validate=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported comparison bundle schema")
    if set(manifest.get("artifacts", {})) != set(PAYLOADS):
        raise ValueError("Incomplete comparison artifact manifest")
    for name in PAYLOADS:
        if file_digest(directory / name) != manifest["artifacts"][name]:
            raise ValueError("Comparison artifact checksum mismatch: " + name)
    bundle = BoundaryBundle(
        manifest,
        {name: read_tsv(directory / name, columns) for name, columns in TABLES.items()},
        *(read_jsonl(directory / name) for name in JSONL_FILES),
    )
    if validate:
        validate_bundle(bundle)
    return bundle


def _nodes(rows):
    result = {}
    for index, row in enumerate(rows):
        code = row["concept_id"]
        if concept_id(code) != code or code in result or row["row_id"] != str(index):
            raise ValueError("Graph has invalid native node indices")
        depth = int(row["root_depth"])
        if depth < 0 or str(depth) != row["root_depth"]:
            raise ValueError("Graph has invalid root depth")
        result[code] = {"label": row["label"], "root_depth": depth}
    return result


def _edges(rows, nodes):
    result = set()
    for row in rows:
        edge = (row["parent_id"], row["child_id"])
        if edge in result or row["relation"] != "is_a":
            raise ValueError("Graph has invalid relation or duplicate edges")
        result.add(edge)
    require_dag(nodes, result)
    return result


def validate_bundle(bundle):
    """Accept a loaded bundle or directory. Raise on incomplete/corrupt structures."""
    if not isinstance(bundle, BoundaryBundle):
        bundle = load_bundle(bundle, validate=False)
    m, tables = bundle.manifest, bundle.tables
    if m.get("side") not in {"keep", "pyhealth"}:
        raise ValueError("Unknown comparison side")
    settings = checked_settings(m["settings"])
    # Older v1 bundles retain their original metadata. Do not infer a new raw
    # mapping policy for a historical bundle that did not declare one.
    if "mapping_contract" in m and m["mapping_contract"] != MAPPING_CONTRACT:
        raise ValueError("Unsupported raw mapping contract")
    if "timestamp_contract" in m and m["timestamp_contract"] != TIMESTAMP_CONTRACT:
        raise ValueError("Unsupported timestamp contract")
    if m["settings"] != settings:
        raise ValueError("All matched settings must be declared")
    if set(m["boundaries"]) != set(BOUNDARIES):
        raise ValueError("All four boundaries require provenance")
    for info in m["boundaries"].values():
        if (
            info.get("provenance_kind")
            not in {"released_keep_code", "keep_reconstruction", "author_artifact", "pyhealth"}
            or info.get("execution_status") != "produced"
        ):
            raise ValueError("Boundary provenance or execution status is missing")
        if (
            not info.get("producer_entry_point")
            or not info.get("executed_source_sha256")
            or "supplied_common_inputs" not in info
        ):
            raise ValueError("Incomplete producer lineage")
    patients = m["patient_ids"]
    if patients != sorted(set(patients)) or any(opaque_id(p) != p for p in patients):
        raise ValueError("Invalid full patient membership")
    retained, uncut = _nodes(tables["concepts.tsv"]), _nodes(tables["uncut_concepts.tsv"])
    expected_retained = {
        c: metadata
        for c, metadata in uncut.items()
        if c != settings["root"] and metadata["root_depth"] <= settings["max_depth"]
    }
    if retained != expected_retained or uncut.get(settings["root"], {}).get("root_depth") != 0:
        raise ValueError("Retained graph does not match the declared uncut depth profile")
    direct, all_direct = (
        _edges(tables["edges.tsv"], retained),
        _edges(tables["uncut_edges.tsv"], uncut),
    )
    if direct != {(a, b) for a, b in all_direct if a in retained and b in retained}:
        raise ValueError("Retained direct edges disagree with the uncut graph")
    links = set()
    for r in tables["closure_links.tsv"]:
        link = (r["ancestor_id"], r["descendant_id"])
        if r["distance"] != "1" or link in links or not set(link) <= uncut.keys():
            raise ValueError("Invalid selected closure overlay")
        links.add(link)
    if settings["path_source"] == "closure":
        require_dag(uncut, all_direct | links)
    rows, columns = m["row_concept_ids"], m["column_concept_ids"]
    if (
        rows != list(retained)
        or len(set(columns)) != len(columns)
        or set(columns) != set(rows)
        or m["matrix_shape"] != [len(rows), len(columns)]
    ):
        raise ValueError("Matrix axes require complete independent native ID maps")
    sources = m["source_concept_ids"]
    if sources != sorted(set(sources), key=int) or any(concept_id(c) != c for c in sources):
        raise ValueError("Invalid source universe")
    statuses = {}
    for row in tables["rollup_status.tsv"]:
        if row["source_id"] in statuses:
            raise ValueError("Duplicate roll-up source status")
        statuses[row["source_id"]] = row
    if list(statuses) != sources:
        raise ValueError("Roll-up statuses do not cover the complete source universe")
    targets = {code: set() for code in sources}
    for pair in tables["rollup.tsv"]:
        source, target = pair["source_id"], pair["target_id"]
        if source not in targets or target not in retained or target in targets[source]:
            raise ValueError("Invalid roll-up source/target pair")
        targets[source].add(target)
    allowed_reasons = {
        "retained",
        "rolled_up",
        "unknown_code",
        "outside_domain",
        "unreachable",
        "no_target",
    }
    for code, row in statuses.items():
        semantic_reason = m["status_mapping"].get(row["reason"])
        if semantic_reason not in allowed_reasons or (
            semantic_reason in {"retained", "rolled_up"}
        ) != bool(targets[code]):
            raise ValueError("Roll-up status disagrees with targets")
        expected_depth = str(uncut[code]["root_depth"]) if code in uncut else ""
        if row["root_depth"] != expected_depth:
            raise ValueError("Source depth disagrees with the uncut graph")
        if code in retained and (targets[code] != {code} or semantic_reason != "retained"):
            raise ValueError("Retained sources require identity mapping")
    native = {}
    observed = set()
    for row in bundle.histories:
        if set(row) != RAW_FIELDS:
            raise ValueError("Raw history must precede roll-up and prediction projection")
        key = (row["patient_id"], opaque_id(row["diagnosis_id"]))
        if key[0] not in patients or key in native:
            raise ValueError("Invalid raw diagnosis membership or multiplicity")
        if row["event_time"] != event_time(row["event_time"]):
            raise ValueError("Noncanonical history time")
        mapped = row["mapped_concepts"]
        if mapped != sorted(set(mapped), key=int) or any(concept_id(c) != c for c in mapped):
            raise ValueError("Invalid mapped concept IDs")
        source = row["source_concept_id"]
        if source is not None and (concept_id(source) != source or mapped != [source]):
            raise ValueError("Canonical source concept precedence was not preserved")
        if bool(mapped) != (row["unmapped_reason"] is None) or row["unmapped_reason"] not in {
            None,
            "missing_source",
            "unsupported_source_format",
            "empty_icd_mapping",
        }:
            raise ValueError("History mapping state disagrees with mapped IDs")
        observed.update(mapped)
        native[key] = row
    if (
        list(native) != sorted(native)
        or m["observed_source_concept_ids"] != sorted(observed, key=int)
        or not observed <= targets.keys()
    ):
        raise ValueError("Observed sources or raw history order are incomplete")
    rolled = {}
    for row in bundle.rolled_histories:
        if set(row) != RAW_FIELDS | {"rolled_up_concepts"}:
            raise ValueError("Invalid rolled history schema")
        key = (row["patient_id"], row["diagnosis_id"])
        if key in rolled or {k: row[k] for k in RAW_FIELDS} != native.get(key):
            raise ValueError("Rolled history changed raw diagnosis fields")
        expected = sorted(set().union(*(targets[c] for c in row["mapped_concepts"])), key=int)
        if row["rolled_up_concepts"] != expected:
            raise ValueError("Diagnosis targets disagree with exported roll-up")
        rolled[key] = row
    if list(rolled) != list(native):
        raise ValueError("Rolled histories must preserve all raw diagnoses")
    presence, triples = count_cooccurrence(bundle.rolled_histories, patients, rows, columns)
    if bundle.patient_presence != presence:
        raise ValueError("Patient presence disagrees with diagnosis occurrences")
    actual = []
    for row in tables["counts.tsv"]:
        converted = {key: int(value) for key, value in row.items()}
        if any(str(converted[k]) != v for k, v in row.items()) or converted["count"] <= 0:
            raise ValueError("Raw matrix requires positive integer sparse triples")
        actual.append(converted)
    if actual != triples:
        raise ValueError("Raw matrix disagrees with full-history patient counts")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "boundary_status": {name: "internally_consistent" for name in BOUNDARIES},
    }


def write_payloads(directory, tables, histories, rolled_histories, presence):
    directory = Path(directory)
    for name, columns in TABLES.items():
        write_tsv(directory / name, columns, tables[name])
    for name, records in zip(JSONL_FILES, (histories, rolled_histories, presence)):
        write_jsonl(directory / name, records)
    return {name: file_digest(directory / name) for name in PAYLOADS}


def write_manifest(directory, manifest):
    write_json(Path(directory) / "manifest.json", manifest)
