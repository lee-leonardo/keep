"""Independent reference correctness and the public ETL interchange contract."""

import csv
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from keep_reference import build, compute, load_bundle, load_config, validate_bundle
from keep_reference.config import checked_settings
from keep_reference.counts import count_cooccurrence
from keep_reference.fixture import generate_fixture
from keep_reference.io import event_time, file_digest
from keep_reference.pipeline import source_identity

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "tests/fixtures/reference_expected.toml").open("rb") as stream:
    EXPECTED = tomllib.load(stream)


@pytest.fixture
def config(tmp_path):
    source = generate_fixture(tmp_path / "source")
    return load_config(source / "reference.toml")


@pytest.fixture
def bundle(config, tmp_path):
    destination = tmp_path / "bundle"
    build(config, destination)
    return destination


def rewrite_input(config, name, change):
    path = config.inputs[name]
    separator = "\t" if name.isupper() else ","
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter=separator)
        columns, rows = reader.fieldnames, list(reader)
    rows = change(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, delimiter=separator, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def test_four_boundary_native_outputs_match_hand_derivation(config, bundle):
    result = compute(config)
    loaded = load_bundle(bundle)
    assert list(result.patient_ids) == EXPECTED["patient_ids"]
    assert len(result.histories) == EXPECTED["unique_diagnoses"]
    assert list(result.ontology.retained_ids) == EXPECTED["concept_ids"]
    assert [result.ontology.depths[c] for c in result.ontology.retained_ids] == EXPECTED["depths"]
    assert [[r["parent_id"], r["child_id"]] for r in loaded.tables["edges.tsv"]] == EXPECTED[
        "retained_edges"
    ]
    assert len(loaded.tables["uncut_concepts.tsv"]) == 11
    assert len(loaded.tables["uncut_edges.tsv"]) == 8
    assert result.rollup_results["900"]["target_ids"] == EXPECTED["X_all_retained"]
    assert result.rollup_results["990"]["target_ids"] == EXPECTED["Y_closure_all_retained"]
    expected_counts = {
        (first, second): EXPECTED["matrix"][i][j]
        for i, first in enumerate(EXPECTED["concept_ids"])
        for j, second in enumerate(EXPECTED["concept_ids"])
        if EXPECTED["matrix"][i][j]
    }
    assert loaded.counts_by_concept() == expected_counts
    assert (
        loaded.manifest["row_concept_ids"]
        == loaded.manifest["column_concept_ids"]
        == EXPECTED["concept_ids"]
    )
    assert any(r == {"patient_id": "5", "concept_ids": []} for r in result.patient_presence)
    assert all(not r["concept_ids"] for r in result.patient_presence if r["patient_id"] == "6")
    assert not any("250" in pair for pair in loaded.counts_by_concept())
    assert loaded.counts_by_concept()[("200", "200")] == 1
    assert validate_bundle(loaded)["valid"]


def test_raw_history_is_not_rolled_or_prediction_history(config):
    result = compute(config)
    assert not any(r["patient_id"] == "5" for r in result.histories)
    p6 = {r["diagnosis_id"]: r for r in result.histories if r["patient_id"] == "6"}
    assert p6["601"]["source_concept_id"] == "300"
    assert p6["601"]["mapped_concepts"] == ["300"]  # Nonzero canonical ID beats BAD source.
    assert p6["601"]["source_value"] == "BAD:ignored"
    assert p6["601"]["event_time"] == "2020-01-03T01:02:03.123456"
    assert p6["602"]["event_time"] == "2020-01-03T00:00:00.000000"  # Date fallback.
    assert p6["603"]["event_time"] is None and p6["603"]["visit_id"] is None
    assert p6["606"]["unmapped_reason"] == "empty_icd_mapping"
    assert p6["607"]["unmapped_reason"] == "missing_source"
    assert p6["608"]["unmapped_reason"] == "unsupported_source_format"
    assert not any("rolled_up_concepts" in r for r in result.histories)
    for code, reason in [
        ("600", "outside_domain"),
        ("700", "unreachable"),
        ("800", "unknown_code"),
        ("999999", "unknown_code"),
    ]:
        assert result.rollup_results[code]["reason"] == reason
        assert result.rollup_results[code]["target_ids"] == []
    assert all(r["mapped_concepts"] == ["900"] for r in result.histories if r["patient_id"] == "1")
    assert len([r for r in result.histories if r["patient_id"] == "3"]) == 1


def test_policy_and_traversal_discriminators(config):
    baseline = compute(config)
    first = compute(replace(config, settings=checked_settings({"target_policy": "first_retained"})))
    direct = compute(replace(config, settings=checked_settings({"path_source": "direct"})))
    assert first.rollup_results["900"]["target_ids"] == EXPECTED["X_first_retained"]
    assert direct.rollup_results["990"]["target_ids"] == EXPECTED["Y_direct"]
    assert direct.rollup_results["990"]["reason"] == "no_target"
    assert (
        baseline.ontology.direct_edges
        == first.ontology.direct_edges
        == direct.ontology.direct_edges
    )
    assert ("200", "990") not in baseline.ontology.direct_edges
    assert ("200", "990", 1) in baseline.ontology.closure_links


def test_no_shared_algorithms_or_pyhealth_available_in_reference_process(config, tmp_path):
    # -S removes site packages; an import guard also rejects both PyHealth and
    # the old PyHealth-backed preparation module during the independent build.
    code = """import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'pyhealth','keep_preprocessing','torch','numpy','polars','pandas'}:
   raise ImportError('blocked dependency')
sys.meta_path.insert(0, Block())
sys.argv=['keep_reference','build','--config',sys.argv[1],'--output',sys.argv[2]]
runpy.run_module('keep_reference', run_name='__main__')
"""
    # Config object stores paths, but the process intentionally opens the same
    # explicit on-disk config, rather than receiving a produced graph or matrix.
    path = config.inputs["person"].parents[1] / "reference.toml"
    child = subprocess.run(
        [sys.executable, "-S", "-c", code, str(path), str(tmp_path / "independent")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["provenance_kind"] == "keep_reconstruction"
    assert load_bundle(tmp_path / "independent").counts_by_concept()


def test_provenance_fingerprints_sources_inputs_and_missing_author_coverage(config, bundle):
    m = load_bundle(bundle).manifest
    assert m["schema_version"] == "keep-comparison-v1"
    assert m["inputs"] == {name: file_digest(path) for name, path in config.inputs.items()}
    assert m["implementation"]["source_sha256"] == source_identity()["source_sha256"]
    for relative, digest in m["implementation"]["executed_source_files"].items():
        assert file_digest(ROOT / relative) == digest
    assert m["environment"]["dependency_profile"] == "python_standard_library_only"
    for name, info in m["boundaries"].items():
        assert info["provenance_kind"] == "keep_reconstruction"
        assert info["supplied_common_inputs"] == []
        assert info["execution_status"] == "produced"
        assert m["author_reference_coverage"][name] == "missing_reference"
        assert all(m["artifacts"][file] == digest for file, digest in info["output_sha256"].items())
    assert str(ROOT) not in (bundle / "manifest.json").read_text()
    assert str(config.inputs["person"]) not in (bundle / "manifest.json").read_text()


def test_row_permutation_hash_seed_and_recomputation(config, tmp_path):
    source = generate_fixture(tmp_path / "shuffled", seed=71)
    outputs = []
    for name, path, seed in [
        ("first", config.inputs["person"].parents[1] / "reference.toml", "1"),
        ("shuffled", source / "reference.toml", "101"),
        ("repeated", source / "reference.toml", "73"),
    ]:
        destination = tmp_path / (name + "-bundle")
        child = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "keep_reference",
                "build",
                "--config",
                str(path),
                "--output",
                str(destination),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert child.returncode == 0, child.stderr
        outputs.append(load_bundle(destination))
    assert (
        outputs[0].manifest["artifacts"]
        == outputs[1].manifest["artifacts"]
        == outputs[2].manifest["artifacts"]
    )
    assert outputs[0].manifest["inputs"] != outputs[1].manifest["inputs"]
    assert outputs[1].manifest == outputs[2].manifest


def test_conditional_counter_explicit_shared_inputs_and_two_diagnosis_rule():
    one = {
        "patient_id": "1",
        "diagnosis_id": "1",
        "source_concept_id": "900",
        "visit_id": "v1",
        "rolled_up_concepts": ["100", "300", "100"],
    }
    two = one | {"diagnosis_id": "2", "source_concept_id": "901", "visit_id": "v2"}
    records = [
        one,
        two,
        one | {"patient_id": "2"},
        two | {"patient_id": "2", "visit_id": "v1"},
        one | {"patient_id": "3"},
        one | {"patient_id": "3"},
    ]
    presence, triples = count_cooccurrence(records, ["1", "2", "3", "4"], ["100", "200", "300"])
    assert triples == [
        {"row": 0, "column": 0, "count": 2},
        {"row": 0, "column": 2, "count": 2},
        {"row": 2, "column": 0, "count": 2},
        {"row": 2, "column": 2, "count": 2},
    ]
    assert presence[-1] == {"patient_id": "4", "concept_ids": []}
    with pytest.raises(ValueError, match="Conflicting"):
        count_cooccurrence([one, one | {"visit_id": "different"}], ["1"], ["100", "300"])


def test_honest_independent_axis_permutations_decode_correctly(config, bundle):
    data = load_bundle(bundle)
    rows = EXPECTED["concept_ids"][::-1]
    columns = EXPECTED["concept_ids"][2:] + EXPECTED["concept_ids"][:2]
    presence, triples = count_cooccurrence(
        data.rolled_histories, data.manifest["patient_ids"], rows, columns
    )
    nodes = {r["concept_id"]: r for r in data.tables["concepts.tsv"]}
    data.tables["concepts.tsv"] = [nodes[code] | {"row_id": str(i)} for i, code in enumerate(rows)]
    data.tables["counts.tsv"] = [{k: str(v) for k, v in t.items()} for t in triples]
    data.manifest["row_concept_ids"], data.manifest["column_concept_ids"] = rows, columns
    assert validate_bundle(data)["valid"]
    assert data.counts_by_concept() == load_bundle(bundle).counts_by_concept()
    data.tables["counts.tsv"][0]["count"] = "99"
    with pytest.raises(ValueError, match="Raw matrix"):
        validate_bundle(data)


def test_missing_axis_or_wrong_id_value_association_is_rejected(bundle):
    data = load_bundle(bundle)
    data.manifest["column_concept_ids"].pop()
    with pytest.raises(ValueError, match="axes"):
        validate_bundle(data)
    data = load_bundle(bundle)
    columns = data.manifest["column_concept_ids"]
    columns[0], columns[5] = columns[5], columns[0]
    # Full coverage and shape remain valid; the values now refer to wrong IDs.
    with pytest.raises(ValueError, match="Raw matrix"):
        validate_bundle(data)


def test_missing_boundary_is_not_success_and_native_status_mapping_survives(bundle):
    data = load_bundle(bundle)
    data.manifest["boundaries"]["rollup"]["execution_status"] = "missing_reference"
    with pytest.raises(ValueError, match="provenance"):
        validate_bundle(data)
    data = load_bundle(bundle)
    data.manifest["status_mapping"]["native_identity"] = "retained"
    for row in data.tables["rollup_status.tsv"]:
        if row["reason"] == "retained":
            row["reason"] = "native_identity"
    assert validate_bundle(data)["valid"]
    assert any(r["reason"] == "native_identity" for r in data.tables["rollup_status.tsv"])


@pytest.mark.parametrize(
    "artifact",
    ["histories.jsonl", "edges.tsv", "rollup.tsv", "counts.tsv", "patient_presence.jsonl"],
)
def test_changed_boundary_artifact_fails_validation(bundle, artifact):
    data = load_bundle(bundle)
    if artifact == "histories.jsonl":
        data.histories[0]["visit_id"] = "changed"
    elif artifact == "edges.tsv":
        data.tables[artifact].pop()
    elif artifact == "rollup.tsv":
        # Remove a deep-source target, leaving the diagnosis targets intact.
        data.tables[artifact] = [
            r
            for r in data.tables[artifact]
            if not (r["source_id"] == "900" and r["target_id"] == "100")
        ]
    elif artifact == "patient_presence.jsonl":
        data.patient_presence[0]["concept_ids"].pop()
    else:
        data.tables[artifact][0]["count"] = "3"
    with pytest.raises(ValueError):
        validate_bundle(data)
    # Independent transport-integrity guard also catches untouched manifests.
    with (bundle / artifact).open("a") as stream:
        stream.write("corrupt\n")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle)


@pytest.mark.parametrize("table", ["CONCEPT", "CONCEPT_ANCESTOR", "CONCEPT_RELATIONSHIP"])
def test_vocabulary_zero_sentinel_does_not_change_reference_outputs(config, tmp_path, table):
    original = build(config, tmp_path / "original")
    sentinel_rows = {
        "CONCEPT": [
            {
                "concept_id": "0",
                "concept_name": "No matching concept",
                "domain_id": "Metadata",
                "vocabulary_id": "None",
                "concept_class_id": "Undefined",
                "standard_concept": "",
                "concept_code": "No matching concept",
                "invalid_reason": "",
            }
        ],
        "CONCEPT_ANCESTOR": [
            {
                "ancestor_concept_id": "0",
                "descendant_concept_id": "0",
                "min_levels_of_separation": "0",
                "max_levels_of_separation": "0",
            }
        ],
        "CONCEPT_RELATIONSHIP": [
            {
                "concept_id_1": "9001",
                "concept_id_2": "0",
                "relationship_id": "Maps to",
                "invalid_reason": "",
            },
            {
                "concept_id_1": "0",
                "concept_id_2": "10",
                "relationship_id": "Is a",
                "invalid_reason": "",
            },
        ],
    }
    rewrite_input(config, table, lambda rows: rows + sentinel_rows[table])
    with_sentinel = build(config, tmp_path / "with-sentinel")
    assert with_sentinel.manifest["inputs"][table] != original.manifest["inputs"][table]
    # The sentinel must not enter graphs, observed mapping IDs, roll-up or counts.
    assert with_sentinel.manifest["artifacts"] == original.manifest["artifacts"]


def test_raw_reader_preserves_opaque_ids_and_rejects_conflicts(config):
    def modify(rows):
        p1 = next(r for r in rows if r["person_id"] == "1")
        p1["condition_occurrence_id"] = "00001"
        rows.append(dict(p1))
        return rows

    rewrite_input(config, "condition_occurrence", modify)
    assert any(r["diagnosis_id"] == "00001" for r in compute(config).histories)

    def conflict(rows):
        original = next(r for r in rows if r["condition_occurrence_id"] == "00001")
        rows.append(original | {"visit_occurrence_id": "changed"})
        return rows

    rewrite_input(config, "condition_occurrence", conflict)
    with pytest.raises(ValueError, match="Conflicting source"):
        compute(config)


def test_missing_time_fallback_and_invalid_datetime_never_silently_truncated(config):
    def broken(rows):
        rows[0]["condition_start_datetime"] = "not-a-time"
        rows[0]["condition_start_date"] = "2020-01-01"
        return rows

    rewrite_input(config, "condition_occurrence", broken)
    with pytest.raises(ValueError, match="Invalid ISO"):
        compute(config)


@pytest.mark.parametrize(
    "change",
    [
        {"typo": 5},
        {"presence_threshold": 1},
        {"max_depth": True},
        {"path_source": "unknown"},
        {"include_root": True},
    ],
)
def test_unknown_or_unsupported_settings_are_not_silently_ignored(change):
    with pytest.raises(ValueError):
        checked_settings(change)


def test_explicit_cohort_empty_cohort_and_zero_vocabulary(config, tmp_path):
    only = compute(replace(config, patient_ids=("1", "5")))
    assert only.patient_ids == ("1", "5") and all(t["count"] == 1 for t in only.counts)
    assert only.patient_presence[-1]["concept_ids"] == []
    empty = build(replace(config, patient_ids=()), tmp_path / "empty")
    assert not empty.histories and not empty.patient_presence and not empty.tables["counts.tsv"]
    zero = build(replace(config, settings=checked_settings({"max_depth": 0})), tmp_path / "zero")
    assert zero.manifest["matrix_shape"] == [0, 0] and not zero.tables["counts.tsv"]
    with pytest.raises(ValueError, match="unknown patient"):
        compute(replace(config, patient_ids=("missing",)))


def test_invalid_graph_cycle_is_an_error(config):
    def cycle(rows):
        rows.append(
            {
                "concept_id_1": "10",
                "concept_id_2": "900",
                "relationship_id": "Is a",
                "invalid_reason": "",
            }
        )
        return rows

    rewrite_input(config, "CONCEPT_RELATIONSHIP", cycle)
    with pytest.raises(ValueError, match="cycle"):
        compute(config)


def test_refuse_overwrite_and_wrong_source_checkout(config, bundle, tmp_path):
    before = file_digest(bundle / "manifest.json")
    with pytest.raises(ValueError, match="fresh"):
        build(config, bundle)
    assert file_digest(bundle / "manifest.json") == before
    with pytest.raises(ValueError, match="different KEEP"):
        build(config, tmp_path / "wrong", expected_checkout=tmp_path)
    with pytest.raises(ValueError, match="revision"):
        build(replace(config, expected_keep_head="0" * 40), tmp_path / "wrong-head")


def test_cli_failure_does_not_log_paths_or_source_identifiers(config, tmp_path):
    path = config.inputs["person"].parents[1] / "reference.toml"

    def broken(rows):
        rows[0]["condition_start_datetime"] = "SECRET_SOURCE_IDENTIFIER"
        return rows

    rewrite_input(config, "condition_occurrence", broken)
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "keep_reference",
            "build",
            "--config",
            str(path),
            "--output",
            str(tmp_path / "failed"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert not (tmp_path / "failed").exists()
    assert str(tmp_path) not in child.stderr and "SECRET_SOURCE_IDENTIFIER" not in child.stderr
    assert not child.stdout and json.loads(child.stderr)["status"] == "error"


@pytest.mark.parametrize("source_id,reason", [("9001", "D"), ("9002", "U")])
def test_retired_icd_sources_keep_valid_historical_mappings(config, source_id, reason):
    def retire(rows):
        for row in rows:
            if row["concept_id"] == source_id:
                row["invalid_reason"] = reason
        return rows

    rewrite_input(config, "CONCEPT", retire)
    result = compute(config)
    first_patient = [row for row in result.histories if row["patient_id"] == "1"]
    assert [row["mapped_concepts"] for row in first_patient] == [["900"], ["900"]]
    assert all(row["unmapped_reason"] is None for row in first_patient)
    assert (
        next(p for p in result.patient_presence if p["patient_id"] == "1")["concept_ids"]
        == EXPECTED["X_all_retained"]
    )


def test_retired_source_does_not_revive_an_invalid_maps_to_relationship(config):
    def invalidate_link(rows):
        for row in rows:
            if row["concept_id_1"] == "9001" and row["relationship_id"] == "Maps to":
                row["invalid_reason"] = "D"
        return rows

    rewrite_input(config, "CONCEPT_RELATIONSHIP", invalidate_link)
    first = next(
        row
        for row in compute(config).histories
        if row["patient_id"] == "1" and row["diagnosis_id"] == "1"
    )
    assert first["mapped_concepts"] == [] and first["unmapped_reason"] == "empty_icd_mapping"


def test_raw_mapping_preserves_targets_for_later_rollup_classification(config):
    baseline_counts = compute(config).counts

    def more_concepts(rows):
        template = dict(rows[0])
        for code, vocabulary, standard in [("810", "SNOMED", ""), ("820", "LOINC", "S")]:
            rows.append(
                template
                | {
                    "concept_id": code,
                    "concept_code": code,
                    "concept_name": "Synthetic target",
                    "vocabulary_id": vocabulary,
                    "standard_concept": standard,
                    "domain_id": "Condition",
                    "invalid_reason": "",
                }
            )
        return rows

    rewrite_input(config, "CONCEPT", more_concepts)

    def more_mappings(rows):
        rows.extend(
            {
                "concept_id_1": "9001",
                "concept_id_2": code,
                "relationship_id": "Maps to",
                "invalid_reason": "",
            }
            for code in ["600", "800", "810", "820", "999999"]
        )
        return rows

    rewrite_input(config, "CONCEPT_RELATIONSHIP", more_mappings)
    result = compute(config)
    first = next(
        row for row in result.histories if row["patient_id"] == "1" and row["diagnosis_id"] == "1"
    )
    assert first["mapped_concepts"] == ["600", "800", "810", "820", "900", "999999"]
    assert first["unmapped_reason"] is None
    for code, reason in [
        ("600", "outside_domain"),
        ("800", "unknown_code"),
        ("810", "unknown_code"),
        ("820", "unknown_code"),
        ("999999", "unknown_code"),
    ]:
        assert result.rollup_results[code]["reason"] == reason
        assert result.rollup_results[code]["target_ids"] == []
    rolled = next(
        row
        for row in result.rolled_histories
        if row["patient_id"] == "1" and row["diagnosis_id"] == "1"
    )
    assert rolled["rolled_up_concepts"] == EXPECTED["X_all_retained"]
    assert result.counts == baseline_counts


def test_mapping_and_precision_contracts_are_declared_without_changing_settings(bundle):
    from keep_reference.config import DEFAULT_SETTINGS, MAPPING_CONTRACT, TIMESTAMP_CONTRACT

    data = load_bundle(bundle)
    assert data.manifest["settings"] == DEFAULT_SETTINGS
    assert data.manifest["mapping_contract"] == MAPPING_CONTRACT
    assert data.manifest["timestamp_contract"] == TIMESTAMP_CONTRACT
    data.manifest["mapping_contract"]["target_eligibility"] = "silently_filter_targets"
    with pytest.raises(ValueError, match="mapping contract"):
        validate_bundle(data)


@pytest.mark.parametrize("fraction", [".1234561", ".1234569", ".1234560", ",1234561"])
def test_finer_than_microsecond_input_is_rejected_before_parsing(fraction):
    with pytest.raises(ValueError, match="microsecond"):
        event_time("2020-01-01T00:00:00" + fraction, "2000-01-01")


@pytest.mark.parametrize(
    "fraction,expected", [(".123456", "123456"), (".1", "100000"), (",123456", "123456")]
)
def test_supported_fractional_seconds_are_preserved(fraction, expected):
    assert event_time("2020-01-01T00:00:00" + fraction) == "2020-01-01T00:00:00." + expected


@pytest.mark.parametrize("fraction", [".1234561", ".1234569"])
def test_csv_build_rejects_unsupported_precision_without_publishing_bundle(
    config, tmp_path, fraction
):
    def precise(rows):
        rows[0]["condition_start_datetime"] = "2020-01-01T00:00:00" + fraction
        rows[0]["condition_start_date"] = "2000-01-01"
        return rows

    rewrite_input(config, "condition_occurrence", precise)
    output = tmp_path / "unsupported-precision"
    with pytest.raises(ValueError, match="microsecond"):
        build(config, output)
    assert not output.exists()
