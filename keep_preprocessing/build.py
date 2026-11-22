"""Adapt explicit local tables through public PyHealth APIs into one bundle."""

import csv
import gzip
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from . import KEEP_BASE, SCHEMA_VERSION
from .bundle import (
    PAYLOADS,
    digest,
    object_digest,
    read_jsonl,
    validate,
    write_json,
    write_jsonl,
    write_table,
)
from .projection import project, timestamp
from .runtime import check_runtime

SOURCE_FILES = {
    "omop": ["person.csv", "visit_occurrence.csv", "death.csv", "condition_occurrence.csv"],
    "mimic4": [
        "hosp/patients.csv.gz",
        "hosp/admissions.csv.gz",
        "icu/icustays.csv.gz",
        "hosp/diagnoses_icd.csv.gz",
    ],
}
VOCABULARY_FILES = ["CONCEPT.csv", "CONCEPT_RELATIONSHIP.csv", "CONCEPT_ANCESTOR.csv"]


def omop_config():
    def table(file, patient, time, attrs):
        return {
            "file_path": file,
            "patient_id": patient,
            "timestamp": time,
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "attributes": attrs,
        }

    return {
        "version": "5.3",
        "tables": {
            "person": table(
                "person.csv", "person_id", None, ["year_of_birth", "month_of_birth", "day_of_birth"]
            ),
            "visit_occurrence": table(
                "visit_occurrence.csv",
                "person_id",
                "visit_start_datetime",
                ["visit_occurrence_id", "visit_start_datetime", "visit_end_datetime"],
            ),
            "death": table("death.csv", "person_id", "death_datetime", []),
            "condition_occurrence": table(
                "condition_occurrence.csv",
                "person_id",
                "condition_start_datetime",
                [
                    "condition_occurrence_id",
                    "condition_concept_id",
                    "condition_source_value",
                    "visit_occurrence_id",
                ],
            ),
        },
    }


def text(value):
    return None if value is None or str(value).strip() == "" else str(value).strip()


def omop_history(patient, mapper):
    """Preserve occurrence identity and canonical-ID precedence at the fork boundary."""
    records = {}
    for event in patient.get_events(event_type="condition_occurrence"):
        identity = text(event.condition_occurrence_id)
        if identity is None:
            raise ValueError("OMOP condition_occurrence_id is required")
        concept = text(event.condition_concept_id)
        source = text(event.condition_source_value)
        if concept is not None:
            concept = str(int(concept))
            if int(concept) < 0:
                raise ValueError("OMOP concept IDs must be nonnegative integers")
        concept = None if concept in {None, "0"} else concept
        if concept:
            mapped, reason = [concept], None
        elif source is None:
            mapped, reason = [], "missing_source"
        else:
            prefix, separator, code = source.partition(":")
            version = {"ICD9CM": 9, "ICD10CM": 10}.get(prefix)
            if not separator or version is None or not code.strip():
                mapped, reason = [], "unsupported_source_format"
            else:
                mapped = sorted({str(c) for c in mapper.map(code.strip(), version)}, key=int)
                reason = None if mapped else "empty_icd_mapping"
        record = {
            "diagnosis_id": identity,
            "visit_id": text(event.visit_occurrence_id),
            "timestamp": event.timestamp,
            "source_value": source,
            "source_concept": concept,
            "concept_ids": mapped,
            "unmapped_reason": reason,
        }
        if records.setdefault(identity, record) != record:
            raise ValueError("OMOP contains conflicting copies of a diagnosis")
    return {"patient_id": patient.patient_id, "conditions": list(records.values())}


def read_cohort(path):
    if path is None:
        return {}
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["patient_id", "label", "index_date"]:
            raise ValueError("Cohort CSV requires patient_id,label,index_date")
        for row in reader:
            patient = row["patient_id"]
            if not patient or patient in result or row["label"] not in {"0", "1"}:
                raise ValueError("Cohort requires unique patients and binary labels")
            label, date = int(row["label"]), timestamp(row["index_date"])
            if label == 1 and date is None:
                raise ValueError("Positive cases require an index date")
            if label == 0 and date is not None:
                raise ValueError(
                    "This protocol uses uncensored controls; leave their index_date empty"
                )
            result[patient] = (label, date)
    return result


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def has_data_rows(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Source tables require a header")
        return next(reader, None) is not None


def build(
    *,
    input_format,
    source,
    vocabulary,
    vocabulary_release,
    output,
    cache_dir,
    cohort=None,
    fit_patients=None,
    target_policy="all_retained",
    path_source="closure",
    max_depth=5,
):
    import polars as pl
    from pyhealth import datasets
    from pyhealth.medcode import OMOPVocabulary
    from pyhealth.medcode.omop import ICDToSNOMEDMapper
    from pyhealth.tasks import KeepConditionHistoryTask, build_keep_omop_ancestor_rollup

    if not vocabulary_release.strip():
        raise ValueError("An explicit vocabulary release is required")
    output, source, vocabulary = (
        Path(output).resolve(),
        Path(source).resolve(),
        Path(vocabulary).resolve(),
    )
    if output.exists():
        raise ValueError("Output already exists; select a fresh bundle directory")
    runtime, environment = check_runtime()
    parameters = {
        "root": "4274025",
        "max_depth": max_depth,
        "depth_measure": "min",
        "target_policy": target_policy,
        "path_source": path_source,
        "include_root": False,
        "orphan_rescue": False,
        "presence_threshold": 2,
        "threshold_stage": "post_rollup",
        "diagonal": "patient_prevalence",
        "history_window": "complete",
        "count_unit": "patient",
        "mimic4_time": "admission",
    }
    fingerprints = {"source/" + name: digest(source / name) for name in SOURCE_FILES[input_format]}
    for name in VOCABULARY_FILES + (
        ["VOCABULARY.csv"] if (vocabulary / "VOCABULARY.csv").is_file() else []
    ):
        fingerprints["vocabulary/" + name] = digest(vocabulary / name)
    if cohort:
        fingerprints["projection_cohort"] = digest(cohort)
    if fit_patients:
        fingerprints["fit_patients"] = digest(fit_patients)
    cohort_spec = read_cohort(cohort)
    cache_key = object_digest(
        {
            "inputs": fingerprints,
            "parameters": parameters,
            "pyhealth": runtime,
            "format": input_format,
            "vocabulary_release": vocabulary_release,
            "omop_config": omop_config(),
            "schema": SCHEMA_VERSION,
        }
    )
    cache = Path(cache_dir).resolve() / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    vocab = OMOPVocabulary(vocabulary)
    processor = build_keep_omop_ancestor_rollup(
        vocab,
        target_policy=target_policy,
        include_root=False,
        max_depth=max_depth,
        on_unmapped="empty",
        path_source=path_source,
    )
    ids = sorted(processor.retained_codes, key=int)
    hierarchy = vocab.condition_hierarchy(
        max_depth=None, depth_measure="min", connect_orphans=False
    )
    depth_by_id = {
        str(code): int(depth)
        for code, depth in hierarchy.nodes.select("concept_id", "depth").iter_rows()
    }
    concept_table = vocab.scan("CONCEPT").collect(engine="streaming")
    labels = {
        str(code): name
        for code, name in concept_table.select("concept_id", "concept_name").iter_rows()
    }
    if concept_table["concept_id"].n_unique() != len(concept_table):
        raise ValueError("Vocabulary concept IDs must be unique")
    valid = concept_table.filter(
        (pl.col("invalid_reason").fill_null("").str.strip_chars() == "")
        & (pl.col("standard_concept") == "S")
        & (pl.col("domain_id") == "Condition")
    ).select("concept_id")
    mapper = ICDToSNOMEDMapper(vocab.icd_mapper().table.join(valid, on="concept_id", how="semi"))
    task = KeepConditionHistoryTask(
        mimic_version=4, vocabulary=vocab, vocabulary_release=vocabulary_release
    )
    empty_cohort = not has_data_rows(source / SOURCE_FILES[input_format][0])
    if empty_cohort and any(
        has_data_rows(source / name) for name in SOURCE_FILES[input_format][1:]
    ):
        raise ValueError("An empty patient table cannot have other patient events")
    if empty_cohort:
        # The pinned reader's cache writer requires a nonempty event schema.
        # An explicit empty input has a well-defined bundle without invoking it.
        samples, patient_ids = [], []
    elif input_format == "omop":
        config = cache / "omop_keep.yaml"
        write_json(config, omop_config())
        dataset = datasets.OMOPDataset(
            root=str(source),
            tables=["condition_occurrence"],
            config_path=str(config),
            cache_dir=str(cache / "dataset"),
            num_workers=1,
        )
    else:
        config = Path(datasets.__file__).parent / "configs" / "mimic4_keep.yaml"
        dataset = datasets.MIMIC4Dataset(
            ehr_root=str(source),
            ehr_tables=["diagnoses_icd"],
            ehr_config_path=str(config),
            cache_dir=str(cache / "dataset"),
            num_workers=1,
        )
        samples = dataset.set_task(task)
    # Sorting patients is needed for stable export. Dataset samples are cached;
    # only patient indices are sorted, not all of their diagnosis event payloads.
    if not empty_cohort and input_format == "omop":
        patient_ids = sorted(str(p) for p in dataset.unique_patient_ids)
        samples = (omop_history(dataset.get_patient(p), mapper) for p in patient_ids)
    elif not empty_cohort:
        index = sorted((str(sample["patient_id"]), i) for i, sample in enumerate(samples))
        patient_ids = [p for p, _ in index]
        # Keep the cache-backed sample object separate from the ordered generator.
        cached_samples = samples
        samples = (cached_samples[i] for _, i in index)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Dataset contains duplicate patient histories")
    fitting = set(patient_ids)
    if fit_patients is not None:
        selected = Path(fit_patients).read_text().splitlines()
        if (
            any(not p for p in selected)
            or len(selected) != len(set(selected))
            or not set(selected) <= fitting
        ):
            raise ValueError("Fitting selection must list distinct known patient IDs")
        fitting = set(selected)
    if not set(cohort_spec) <= set(patient_ids):
        raise ValueError("Projection cohort contains unknown patients")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_table(
            stage,
            "concepts.tsv",
            [(i, code, labels[code], depth_by_id[code]) for i, code in enumerate(ids)],
        )
        edges = sorted(
            {
                (str(p), str(c))
                for p, c in hierarchy.edges.select("parent_id", "child_id").iter_rows()
                if str(p) in processor.retained_codes and str(c) in processor.retained_codes
            },
            key=lambda e: tuple(map(int, e)),
        )
        write_table(stage, "edges.tsv", edges)
        # Include failure classifications as well as reachable sources.
        universe = set(
            str(code)
            for code in concept_table.filter(
                (pl.col("vocabulary_id") == "SNOMED")
                & (pl.col("standard_concept") == "S")
                & (pl.col("invalid_reason").fill_null("").str.strip_chars() == "")
            )["concept_id"]
        )
        mappings = {r.source: r for r in processor.rollup_codes(sorted(universe, key=int))}
        stats = Counter()
        losses = Counter()

        def history_rows():
            for sample in samples:
                for condition in sorted(sample["conditions"], key=lambda r: r["diagnosis_id"]):
                    codes = sorted(set(condition["concept_ids"]), key=int)
                    for code in codes:
                        if code not in mappings:
                            mappings[code] = processor.rollup_codes([code])[0]
                    targets = sorted(
                        {target for code in codes for target in mappings[code].targets}, key=int
                    )
                    reason = condition["unmapped_reason"] or (
                        "mapped" if targets else "empty_rollup"
                    )
                    stats["diagnoses"] += 1
                    if reason != "mapped":
                        losses[reason] += 1
                    yield {
                        "patient_id": str(sample["patient_id"]),
                        "diagnosis_id": condition["diagnosis_id"],
                        "visit_id": condition["visit_id"],
                        "event_time": timestamp(condition["timestamp"]),
                        "source_value": condition["source_value"],
                        "mapped_concepts": codes,
                        "rolled_up_concepts": targets,
                        "mapping_outcome": reason,
                    }

        write_jsonl(stage / "histories.jsonl", history_rows())
        write_table(
            stage,
            "rollup.tsv",
            [
                (code, target)
                for code in sorted(mappings, key=int)
                for target in sorted(mappings[code].targets, key=int)
            ],
        )
        write_table(
            stage,
            "rollup_status.tsv",
            [
                (
                    code,
                    "" if mappings[code].source_depth is None else mappings[code].source_depth,
                    mappings[code].reason,
                )
                for code in sorted(mappings, key=int)
            ],
        )

        def normalized_samples():
            current = None
            conditions = []
            for row in read_jsonl(stage / "histories.jsonl"):
                if row["patient_id"] != current:
                    if current in fitting:
                        yield {"patient_id": current, "conditions": conditions}
                    current, conditions = row["patient_id"], []
                conditions.append(
                    {
                        "diagnosis_id": row["diagnosis_id"],
                        "concept_ids": row["mapped_concepts"],
                        "unmapped_reason": None
                        if row["mapping_outcome"] in {"mapped", "empty_rollup"}
                        else row["mapping_outcome"],
                    }
                )
            if current in fitting:
                yield {"patient_id": current, "conditions": conditions}

        result = task.build_cooccurrence(normalized_samples(), rollup=processor, concept_ids=ids)
        matrix = result.matrix.coalesce()
        triples = sorted(
            (i, j, int(n))
            for (i, j), n in zip(matrix.indices().t().tolist(), matrix.values().tolist())
        )
        write_table(stage, "counts.tsv", triples)
        write_jsonl(
            stage / "prediction_histories.jsonl",
            project(read_jsonl(stage / "histories.jsonl"), cohort_spec),
        )
        modules = Path(__file__).parent
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "revisions": {
                "keep_base": KEEP_BASE,
                "keep_head": _git_head(),
                "preprocessing_source_sha256": object_digest(
                    {p.name: digest(p) for p in sorted(modules.glob("*.py"))}
                ),
                "pyhealth": runtime,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.system(),
                "machine": platform.machine(),
                "packages": environment,
                "requirements_lock_sha256": digest(
                    Path(__file__).resolve().parents[1] / "requirements-preprocessing.lock"
                ),
            },
            "input_format": input_format,
            "inputs": fingerprints,
            "vocabulary_release": vocabulary_release,
            "parameters": parameters,
            "graph": {"direction": "parent_to_child", "directed": True, "include_isolates": True},
            "shape": [len(ids), len(ids)],
            "patient_ids": patient_ids,
            "fit_patient_ids": sorted(fitting),
            "projection_cohort": {
                p: {"label": label, "index_date": date}
                for p, (label, date) in sorted(cohort_spec.items())
            },
            "projection_policy": {
                "positive": "strictly_before_index_date",
                "control": "uncensored",
                "ordering": "second_occurrence_then_numeric_concept_id",
                "threshold": 2,
            },
            "statistics": {
                "patients": len(patient_ids),
                "fit_patients": len(fitting),
                "diagnoses": stats["diagnoses"],
                "patients_with_present_targets": result.patients_with_present_targets,
                "mapping_losses": dict(sorted(losses.items())),
            },
            "artifacts": {name: digest(stage / name) for name in PAYLOADS},
        }
        write_json(stage / "manifest.json", manifest)
        validate(stage)
        # Reject source changes while preparation was in progress.
        for name, before in fingerprints.items():
            if name.startswith("source/"):
                path = source / name.removeprefix("source/")
            elif name.startswith("vocabulary/"):
                path = vocabulary / name.removeprefix("vocabulary/")
            else:
                path = Path(cohort if name == "projection_cohort" else fit_patients)
            if digest(path) != before:
                raise ValueError("An input changed while the bundle was being built")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return {"bundle": str(output), "cache_key": cache_key, **validate(output)}
