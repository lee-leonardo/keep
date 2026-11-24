"""An explicit common-CSV configuration; relative paths use the config directory."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .io import concept_id, opaque_id

REQUIRED_INPUTS = {
    "person",
    "visit_occurrence",
    "death",
    "condition_occurrence",
    "CONCEPT",
    "CONCEPT_RELATIONSHIP",
    "CONCEPT_ANCESTOR",
    "VOCABULARY",
}
# Additive metadata: keep the existing settings/config API usable by the ETL
# adapter, while distinguishing raw mapping from graph/roll-up eligibility.
MAPPING_CONTRACT = {
    "version": "keep-reference-raw-mapping-v2",
    "icd_source_validity": "include_retired_sources",
    "maps_to_validity": "blank_invalid_reason",
    "target_eligibility": "all_positive_maps_to_target_ids",
    "target_classification_stage": "rollup",
}
TIMESTAMP_CONTRACT = {
    "precision": "microseconds",
    "excess_fractional_precision": "reject_before_parsing",
    "timezone": "naive_only",
}
DEFAULT_SETTINGS = {
    "root": "4274025",
    "max_depth": 5,
    "depth_measure": "min",
    "target_policy": "all_retained",
    "path_source": "closure",
    "include_root": False,
    "orphan_rescue": False,
    "concept_filter": "active_standard_snomed_condition",
    "edge_direction": "parent_to_child",
    "relation": "is_a",
    "include_isolates": True,
    "source_precedence": "nonzero_canonical_then_icd",
    "icd_normalization": "strip_dots_uppercase",
    "duplicate_policy": "collapse_identical_reject_conflicting",
    "missing_time": "null",
    "event_time": "datetime_then_date",
    "timestamp_precision": "microseconds",
    "patient_universe": "person_table",
    "observation_window": "complete",
    "presence_threshold": 2,
    "threshold_stage": "post_rollup",
    "count_unit": "patient",
    "diagonal": "patient_prevalence",
    "count_one": "keep",
}


def checked_settings(supplied=None):
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict) or not supplied.keys() <= DEFAULT_SETTINGS.keys():
        raise ValueError("Unknown reference setting")
    result = DEFAULT_SETTINGS | supplied
    for key, default in DEFAULT_SETTINGS.items():
        if type(result[key]) is not type(default):
            raise ValueError("Reference setting has the wrong type")
        if key not in {"max_depth", "target_policy", "path_source"} and result[key] != default:
            raise ValueError("Unsupported setting for the common synthetic profile")
    if (
        result["max_depth"] < 0
        or result["target_policy"] not in {"all_retained", "first_retained"}
        or result["path_source"] not in {"closure", "direct"}
    ):
        raise ValueError("Invalid reference roll-up settings")
    return result


@dataclass(frozen=True)
class ReferenceConfig:
    inputs: dict[str, Path]
    vocabulary_release: str
    settings: dict
    patient_ids: tuple[str, ...] | None = None
    source_ids: tuple[str, ...] = ()
    fixture_version: str | None = None
    expected_keep_head: str | None = None


def load_config(path):
    path = Path(path)
    with path.open("rb") as stream:
        data = tomllib.load(stream) if path.suffix == ".toml" else json.load(stream)
    if not isinstance(data, dict) or not data.keys() <= {
        "schema_version",
        "inputs",
        "settings",
        "vocabulary_release",
        "patient_ids",
        "source_ids",
        "fixture_version",
        "expected_keep_head",
    }:
        raise ValueError("Unknown reference configuration field")
    if (
        data.get("schema_version") != "keep-reference-config-v1"
        or not isinstance(data.get("inputs"), dict)
        or set(data.get("inputs", {})) != REQUIRED_INPUTS
    ):
        raise ValueError("Explicit common input table paths and the config schema are required")
    if (
        not isinstance(data.get("vocabulary_release"), str)
        or not data["vocabulary_release"].strip()
    ):
        raise ValueError("An explicit vocabulary release is required")
    inputs = {}
    for name, value in data["inputs"].items():
        if not isinstance(value, str) or not value:
            raise ValueError("Input locations must be explicit nonempty paths")
        inputs[name] = path.parent / value if not Path(value).is_absolute() else Path(value)
    patients = data.get("patient_ids")
    if patients is not None:
        if not isinstance(patients, list):
            raise ValueError("patient_ids must be an explicit list")
        patients = tuple(opaque_id(p) for p in patients)
        if len(set(patients)) != len(patients):
            raise ValueError("Repeated cohort identities")
    sources = data.get("source_ids", [])
    if (
        not isinstance(sources, list)
        or any(concept_id(code) != code for code in sources)
        or len(sources) != len(set(sources))
    ):
        raise ValueError("source_ids must be distinct canonical concept strings")
    return ReferenceConfig(
        inputs,
        data["vocabulary_release"],
        checked_settings(data.get("settings")),
        patients,
        tuple(sources),
        data.get("fixture_version"),
        data.get("expected_keep_head"),
    )
