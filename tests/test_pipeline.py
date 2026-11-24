import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from keep_preprocessing.bundle import digest, read_jsonl, read_table, validate, write_json
from keep_preprocessing.consumers import load_embeddings, load_inputs, save_embeddings
from keep_preprocessing.fixtures import generate
from keep_preprocessing.projection import timestamp
from keep_preprocessing.reference import compare_author_histories, compare_author_projection

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "tests/fixtures/expected.json").read_text())


def run(command, *, seed="1"):
    env = {**os.environ, "PYTHONHASHSEED": seed}
    completed = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240
    )
    assert completed.returncode == 0, completed.stdout[-9000:] + completed.stderr[-9000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    work = tmp_path_factory.mktemp("c0014")
    source = generate(work / "source")
    permuted = generate(work / "permuted", seed=77)
    count_one = generate(work / "count-one", count_one=True)
    empty = generate(work / "empty", empty=True)
    cache = work / "cache"
    results = {}

    def build(name, source_dir=source, fmt="omop", seed="1", *extra):
        path = work / name
        command = [
            sys.executable,
            "-m",
            "keep_preprocessing",
            "build",
            "--input-format",
            fmt,
            "--source",
            str(source_dir / fmt),
            "--vocabulary",
            str(source_dir / "vocabulary"),
            "--vocabulary-release",
            "synthetic-c0014-v1",
            "--cohort",
            str(source_dir / "cohort.csv"),
            "--output",
            str(path),
            "--cache-dir",
            str(cache),
            *extra,
        ]
        results[name] = run(command, seed=seed)
        return path

    paths = {
        "base": build("base"),
        "warm": build("warm", seed="37"),
        "permuted": build("permuted-bundle", permuted, seed="103"),
        "mimic4": build("mimic4-bundle", fmt="mimic4", seed="29"),
        "mimic4_warm": build("mimic4-warm", fmt="mimic4", seed="47"),
        "count_one": build("count-one-bundle", count_one),
        "first_retained": build(
            "first-retained", source, "omop", "19", "--target-policy", "first_retained"
        ),
    }
    changed = work / "changed"
    shutil.copytree(source, changed)
    vocab_file = changed / "vocabulary/VOCABULARY.csv"
    vocab_file.write_text(
        vocab_file.read_text().replace("synthetic-c0014-v1", "synthetic-c0014-v2")
    )
    paths["vocabulary_changed"] = build("vocabulary-changed", changed)
    fitting = work / "fitting.txt"
    fitting.write_text("1\n")
    paths["fitting_only"] = build(
        "fitting-only", source, "omop", "1", "--fit-patients", str(fitting)
    )
    paths["empty_omop"] = build("empty-omop", empty)
    paths["empty_mimic4"] = build("empty-mimic4", empty, "mimic4")
    paths["empty_vocabulary"] = build("empty-vocabulary", source, "omop", "1", "--max-depth", "0")
    return {"paths": paths, "results": results, "work": work, "source": source}


def test_exact_graph_rollup_and_raw_counts(bundles):
    bundle = bundles["paths"]["base"]
    manifest, ids, edges, triples = load_inputs(bundle)
    assert ids == EXPECTED["concept_ids"]
    assert [int(row["root_depth"]) for row in read_table(bundle, "concepts.tsv")] == EXPECTED[
        "depths"
    ]
    assert [[ids[p], ids[c]] for p, c in edges] == EXPECTED["edges"]
    assert EXPECTED["isolate"] not in {ids[i] for edge in edges for i in edge}
    pairs = read_table(bundle, "rollup.tsv")
    assert [r["target_id"] for r in pairs if r["source_id"] == "900"] == EXPECTED["all_retained_X"]
    assert [r["target_id"] for r in pairs if r["source_id"] == "300"] == EXPECTED["retained_A"]
    statuses = {r["source_id"]: r["reason"] for r in read_table(bundle, "rollup_status.tsv")}
    assert statuses["600"] == "outside_domain"
    assert statuses["700"] == "unreachable"
    assert "800" not in statuses  # Deprecated concepts are excluded.
    expected_triples = [
        (i, j, n) for i, row in enumerate(EXPECTED["matrix"]) for j, n in enumerate(row) if n
    ]
    assert triples == expected_triples
    assert manifest["statistics"]["diagnoses"] == EXPECTED["unique_diagnoses"]
    assert (
        list(read_jsonl(bundle / "prediction_histories.jsonl")) == EXPECTED["projected_histories"]
    )
    assert validate(bundle)["valid"]


def test_policy_discriminator_and_cache_invalidation(bundles):
    first = bundles["paths"]["first_retained"]
    assert [
        r["target_id"] for r in read_table(first, "rollup.tsv") if r["source_id"] == "900"
    ] == EXPECTED["first_retained_X"]
    results = bundles["results"]
    assert results["base"]["cache_key"] == results["warm"]["cache_key"]
    assert results["base"]["cache_key"] != results["first-retained"]["cache_key"]
    assert results["base"]["cache_key"] != results["vocabulary-changed"]["cache_key"]
    assert results["base"]["artifact_sha256"] == results["vocabulary-changed"]["artifact_sha256"]


def test_separate_process_reordering_hash_seed_and_warm_cache(bundles):
    paths = bundles["paths"]
    base = json.loads((paths["base"] / "manifest.json").read_text())
    for variant in ["warm", "permuted"]:
        other = json.loads((paths[variant] / "manifest.json").read_text())
        assert base["artifacts"] == other["artifacts"]
        # A warm rebuild has an identical entire manifest; row-permuted input
        # has different source checksums but the same canonical semantics.
        if variant == "warm":
            assert (paths["base"] / "manifest.json").read_bytes() == (
                paths[variant] / "manifest.json"
            ).read_bytes()
        else:
            assert base["inputs"] != other["inputs"]
            for key in [
                "parameters",
                "shape",
                "patient_ids",
                "fit_patient_ids",
                "statistics",
                "projection_cohort",
                "revisions",
            ]:
                assert base[key] == other[key]


def test_formats_agree_and_keep_source_identity(bundles):
    base, mimic = bundles["paths"]["base"], bundles["paths"]["mimic4"]
    for file in [
        "concepts.tsv",
        "edges.tsv",
        "rollup.tsv",
        "rollup_status.tsv",
        "counts.tsv",
        "prediction_histories.jsonl",
    ]:
        assert (base / file).read_bytes() == (mimic / file).read_bytes()
    omop_rows, mimic_rows = (
        list(read_jsonl(base / "histories.jsonl")),
        list(read_jsonl(mimic / "histories.jsonl")),
    )

    def semantic(rows):
        return sorted(
            (
                r["patient_id"],
                r["visit_id"],
                r["event_time"],
                r["source_value"],
                tuple(r["mapped_concepts"]),
                tuple(r["rolled_up_concepts"]),
                r["mapping_outcome"],
            )
            for r in rows
        )

    assert semantic(omop_rows) == semantic(mimic_rows)
    assert {r["diagnosis_id"] for r in omop_rows} == {"1", "2", "3", "4", "5"}
    assert {tuple(json.loads(r["diagnosis_id"])) for r in mimic_rows if r["patient_id"] == "1"} == {
        ("11", "1", "0010", "9"),
        ("12", "1", "0010", "10"),
    }
    assert len([r for r in mimic_rows if r["patient_id"] == "2"]) == 2
    assert len([r for r in mimic_rows if r["patient_id"] == "3"]) == 1
    assert (mimic / "manifest.json").read_bytes() == (
        bundles["paths"]["mimic4_warm"] / "manifest.json"
    ).read_bytes()


def test_empty_inputs_and_vocabulary_keep_explicit_shapes(bundles):
    for name in ["empty_omop", "empty_mimic4"]:
        bundle = bundles["paths"][name]
        result = validate(bundle)
        assert result["shape"] == [7, 7] and result["nonzero_counts"] == 0
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert manifest["patient_ids"] == [] and manifest["statistics"]["diagnoses"] == 0
    result = validate(bundles["paths"]["empty_vocabulary"])
    assert result["shape"] == [0, 0] and result["nonzero_counts"] == 0


def test_count_only_fitting_patients_without_dropping_histories(bundles):
    bundle = bundles["paths"]["fitting_only"]
    manifest, _, _, triples = load_inputs(bundle)
    assert manifest["fit_patient_ids"] == ["1"]
    assert manifest["patient_ids"] == ["1", "2", "3"]
    assert len(list(read_jsonl(bundle / "histories.jsonl"))) == 5
    assert triples == [
        (i, j, 1) for i, row in enumerate(EXPECTED["matrix"]) for j, n in enumerate(row) if n
    ]


def rehash(bundle, name):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["artifacts"][name] = digest(bundle / name)
    write_json(bundle / "manifest.json", manifest)


@pytest.mark.parametrize("kind", ["checksum", "index", "count", "policy", "cycle"])
def test_corruption_and_semantic_tampering_are_rejected(bundles, tmp_path, kind):
    bundle = tmp_path / "bundle"
    shutil.copytree(bundles["paths"]["base"], bundle)
    if kind == "checksum":
        with (bundle / "counts.tsv").open("a") as stream:
            stream.write("0\t0\t99\n")
    elif kind == "index":
        path = bundle / "concepts.tsv"
        lines = path.read_text().splitlines()
        first, second = lines[1].split("\t"), lines[2].split("\t")
        first[1], second[1] = second[1], first[1]
        lines[1], lines[2] = "\t".join(first), "\t".join(second)
        path.write_text("\n".join(lines) + "\n")
        rehash(bundle, "concepts.tsv")
    elif kind == "count":
        path = bundle / "counts.tsv"
        path.write_text(path.read_text().replace("0\t0\t2\n", "0\t0\t3\n"))
        rehash(bundle, "counts.tsv")
    elif kind == "cycle":
        path = bundle / "edges.tsv"
        path.write_text(path.read_text() + "100\t10\n")
        rehash(bundle, "edges.tsv")
    else:
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["parameters"]["presence_threshold"] = 1
        write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError):
        validate(bundle)


def test_embedding_permutation_and_stale_graph_are_rejected(bundles, tmp_path):
    import numpy as np

    bundle = bundles["paths"]["base"]
    ids = EXPECTED["concept_ids"]
    vectors = np.arange(14, dtype=np.float32).reshape(7, 2)
    aligned = tmp_path / "aligned.npz"
    save_embeddings(aligned, vectors, ids, bundle)
    assert np.array_equal(load_embeddings(aligned, ids, bundle), vectors)
    with pytest.raises(ValueError, match="index mismatch"):
        load_embeddings(aligned, ids[::-1], bundle)
    with np.load(aligned) as archive:
        payload = dict(archive)
    payload["edges_sha256"] = np.asarray("stale")
    stale = tmp_path / "stale.npz"
    np.savez(stale, **payload)
    with pytest.raises(ValueError, match="provenance"):
        load_embeddings(stale, ids, bundle)


def test_narrow_matrix_through_public_task_and_counter(bundles):
    from pyhealth.graph import KnowledgeGraph
    from pyhealth.medcode import OMOPVocabulary
    from pyhealth.processors import AncestorRollupProcessor
    from pyhealth.tasks import KeepConditionHistoryTask

    ids = EXPECTED["narrow_axes"]
    graph = KnowledgeGraph(
        triples=[("R", "is_a", code) for code in ids]
        + [(p, "is_a", source) for p in ["100", "300"] for source in ["900", "901"]]
    )
    rollup = AncestorRollupProcessor(
        knowledge_graph=graph,
        domain_by_code=dict.fromkeys(["R", *ids, "900", "901"], "Condition"),
        hierarchy_relation="is_a",
        root="R",
        target_policy="all_retained",
        include_root=False,
        max_depth=1,
    )
    task = KeepConditionHistoryTask(
        mimic_version=4,
        vocabulary=OMOPVocabulary(bundles["source"] / "vocabulary"),
        vocabulary_release="synthetic",
    )
    one = {"diagnosis_id": "1", "concept_ids": ["900"], "unmapped_reason": None, "visit_id": "v1"}
    second = {
        "diagnosis_id": "2",
        "concept_ids": ["901"],
        "unmapped_reason": None,
        "visit_id": "v2",
    }
    histories = [
        {"patient_id": "1", "conditions": [one, second]},
        {"patient_id": "2", "conditions": [one, {**second, "visit_id": "v1"}]},
        {"patient_id": "3", "conditions": [one, one]},
    ]
    result = task.build_cooccurrence(histories, rollup=rollup, concept_ids=ids)
    assert result.matrix.to_dense().tolist() == EXPECTED["narrow_matrix"]
    assert result.unique_diagnosis_events == 5 and result.patients_with_present_targets == 2


def test_public_mimic_task_keeps_same_admission_sequence_different_versions(bundles):
    from types import SimpleNamespace

    from pyhealth.medcode import OMOPVocabulary
    from pyhealth.tasks import KeepConditionHistoryTask

    task = KeepConditionHistoryTask(
        mimic_version=4,
        vocabulary=OMOPVocabulary(bundles["source"] / "vocabulary"),
        vocabulary_release="synthetic",
    )
    events = [
        SimpleNamespace(hadm_id="11", seq_num="1", icd_code="0010", icd_version=version)
        for version in ["9", "10", "9"]
    ]
    patient = SimpleNamespace(
        patient_id="1",
        get_events=lambda event_type: events if event_type == "diagnoses_icd" else [],
    )
    records = task(patient)[0]["conditions"]
    assert len(records) == 2
    assert {tuple(json.loads(r["diagnosis_id"])) for r in records} == {
        ("11", "1", "0010", "9"),
        ("11", "1", "0010", "10"),
    }


def test_count_one_and_raw_preservation(bundles):
    bundle = bundles["paths"]["count_one"]
    triples = {
        (int(r["row"]), int(r["column"])): int(r["count"]) for r in read_table(bundle, "counts.tsv")
    }
    ids = EXPECTED["concept_ids"]
    z = ids.index("200")
    assert triples.pop((z, z)) == EXPECTED["count_one_Z_diagonal"]
    expected = {
        (i, j): n for i, row in enumerate(EXPECTED["matrix"]) for j, n in enumerate(row) if n
    }
    assert triples == expected


def test_exact_author_history_parity(bundles):
    result = compare_author_histories(bundles["paths"]["base"])
    assert result["status"] == "equal"
    assert result["included_patients"] == 1


@pytest.fixture(scope="module")
def history_reference_boundary():
    # A positive condition qualifies before the date; another has its second
    # occurrence exactly at the date. Another case has no qualifying history.
    rows = []
    for patient, events in {
        "case": [
            ("2020-01-01", ["100", "200"]),
            ("2020-01-02", ["100"]),
            ("2020-01-03", ["200"]),
            ("2020-01-04", ["200"]),
        ],
        "control": [("2020-01-01", ["100"]), ("2020-01-04", ["100"])],
        "excluded-case": [("2020-01-03", ["100"]), ("2020-01-04", ["100"])],
    }.items():
        rows.extend(
            {
                "patient_id": patient,
                "diagnosis_id": str(i),
                "event_time": timestamp(date),
                "rolled_up_concepts": targets,
            }
            for i, (date, targets) in enumerate(events)
        )
    cohort = {
        "case": (1, timestamp("2020-01-03")),
        "control": (0, None),
        "excluded-case": (1, timestamp("2020-01-03")),
    }
    from keep_preprocessing.projection import project

    assert project(rows, cohort) == [
        {"patient_id": "case", "label": 1, "concept_ids": ["100"]},
        {"patient_id": "control", "label": 0, "concept_ids": ["100"]},
    ]
    result = compare_author_projection(rows, cohort)
    assert (
        result["status"] == "equal"
        and result["included_positive_patients"] == result["included_negative_patients"] == 1
    )
    return result


def test_author_history_boundary_includes_positive_and_control(history_reference_boundary):
    assert history_reference_boundary["included_patients"] == 2


@pytest.fixture(scope="module")
def consumer_smoke(bundles):
    work = bundles["work"] / "training"
    bundle = bundles["paths"]["count_one"]
    before = digest(bundle / "counts.tsv")
    python = os.environ.get("KEEP_TRAINING_PYTHON", str(ROOT / ".venv/bin/python"))
    n2v_path, glove_path = work / "node2vec.npz", work / "glove.npz"
    n2v = run(
        [
            python,
            "trained_embeddings/our_embeddings/train_node2vec.py",
            "--bundle",
            str(bundle),
            "--output",
            str(n2v_path),
            "--dimensions",
            "8",
            "--walk-length",
            "5",
            "--num-walks",
            "4",
            "--epochs",
            "2",
            "--seed",
            "19",
            "--sample",
            "0",
        ]
    )
    glove = run(
        [
            python,
            "trained_embeddings/our_embeddings/train_glove.py",
            "--bundle",
            str(bundle),
            "--init",
            str(n2v_path),
            "--output",
            str(glove_path),
            "--dimensions",
            "8",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--drop-count-one",
            "--diagonal",
            "drop",
            "--seed",
            "19",
        ]
    )
    assert math.isfinite(n2v["training_loss"]) and n2v["training_loss"] > 0
    assert n2v["nodes"] == 7 and n2v["edges"] == 5 and n2v["isolates"] == 1
    assert math.isfinite(glove["first_loss"]) and math.isfinite(glove["last_loss"])
    assert glove["raw_positive_triples"] == 37 and glove["training_triples"] == 30
    assert before == digest(bundle / "counts.tsv")
    return {"node2vec": n2v, "glove": glove, "raw_counts_unchanged": True}


def test_released_consumers_cpu_smoke(consumer_smoke):
    assert consumer_smoke["node2vec"]["finite_vectors"]
    assert consumer_smoke["glove"]["finite_vectors"]


def test_sanitized_evidence(bundles, consumer_smoke, history_reference_boundary):
    if destination := os.environ.get("KEEP_EVIDENCE_OUTPUT"):
        base = json.loads((bundles["paths"]["base"] / "manifest.json").read_text())
        summary = {
            "input_kind": "generated_synthetic_only",
            "revisions": base["revisions"],
            "environment": base["environment"],
            "fixture_expectations_sha256": digest(ROOT / "tests/fixtures/expected.json"),
            "input_sha256": base["inputs"],
            "payload_hashes": {
                name: json.loads((path / "manifest.json").read_text())["artifacts"]
                for name, path in bundles["paths"].items()
            },
            "checks": {
                "exact_graph_and_rollup": "passed",
                "exact_integer_counts": "passed",
                "source_format_parity": "passed",
                "reordered_input_and_hash_seeds": "passed",
                "warm_cache_identical_manifest": "passed",
                "configuration_and_vocabulary_cache_invalidation": "passed",
            },
            "released_code_history_comparison": compare_author_histories(bundles["paths"]["base"]),
            "released_code_history_boundary_comparison": history_reference_boundary,
            "consumers": consumer_smoke,
            "author_artifact_parity": "not_checked_matched_author_outputs_unavailable",
            "published_result_replication": "not_in_scope",
        }
        write_json(destination, summary)
