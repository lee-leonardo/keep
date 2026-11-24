"""Generate public, wholly synthetic Athena, OMOP and MIMIC-IV inputs."""

import csv
import gzip
import io
import random
from pathlib import Path

ROOT = 4274025


def write_table(path, columns, rows, *, separator=",", seed=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if seed is not None:
        random.Random(seed).shuffle(rows)
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter=separator, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    payload = text.getvalue().encode()
    if path.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    path.write_bytes(payload)


def generate(destination, *, seed=None, count_one=False, empty=False):
    """Write both formats with equivalent mapped diagnoses and admission times."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Fixture destination already exists")
    destination.mkdir(parents=True)
    concepts = [
        (ROOT, "Disease", "Condition", "SNOMED", "S", "64572001"),
        (10, "N1", "Condition", "SNOMED", "S", "10"),
        (20, "N2", "Condition", "SNOMED", "S", "20"),
        (30, "N3", "Condition", "SNOMED", "S", "30"),
        (40, "N4", "Condition", "SNOMED", "S", "40"),
        (100, "B", "Condition", "SNOMED", "S", "100"),
        (200, "Z", "Condition", "SNOMED", "S", "200"),
        (300, "A", "Condition", "SNOMED", "S", "300"),
        (900, "X", "Condition", "SNOMED", "S", "900"),
        (600, "Outside domain", "Observation", "SNOMED", "S", "600"),
        (700, "Unreachable", "Condition", "SNOMED", "S", "700"),
        (800, "Deprecated", "Condition", "SNOMED", "S", "800"),
        (9001, "ICD9 leading zero", "Condition", "ICD9CM", "", "0010"),
        (9002, "ICD10 leading zero", "Condition", "ICD10CM", "", "0010"),
        (9003, "ICD10 I214", "Condition", "ICD10CM", "", "I214"),
        (9004, "ICD10 K1", "Condition", "ICD10CM", "", "K1"),
        (9005, "ICD10 Z1", "Condition", "ICD10CM", "", "Z1"),
        (9006, "ICD10 U1", "Condition", "ICD10CM", "", "U1"),
    ]
    vocab = destination / "vocabulary"
    write_table(
        vocab / "CONCEPT.csv",
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "invalid_reason",
        ],
        [
            (
                i,
                name,
                domain,
                vocabulary,
                "Clinical Finding",
                standard,
                code,
                "D" if i == 800 else "",
            )
            for i, name, domain, vocabulary, standard, code in concepts
        ],
        separator="\t",
        seed=seed,
    )
    direct = [
        (ROOT, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 100),
        (40, 300),
        (100, 900),
        (300, 900),
    ]
    mapping = [
        (9001, 900),
        (9002, 900),
        (9003, 900),
        (9004, 900),
        (9004, 100),
        (9005, 200),
        (9006, 700),
    ]
    write_table(
        vocab / "CONCEPT_RELATIONSHIP.csv",
        ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
        [(child, parent, "Is a", "") for parent, child in direct]
        + [(source, target, "Maps to", "") for source, target in mapping],
        separator="\t",
        seed=seed,
    )
    # The fixture generator derives only input closure rows. Expectations live
    # in tests/fixtures/expected.json and do not call this implementation.
    traversal = direct + [(40, 200)]
    children = {}
    for parent, child in traversal:
        children.setdefault(parent, []).append(child)
    ancestry = []
    for source in [ROOT, 10, 20, 30, 40, 100, 200, 300, 900, 600, 700, 800]:
        distances = {source: [0]}
        pending = [(source, 0)]
        while pending:
            node, depth = pending.pop()
            for child in children.get(node, []):
                distances.setdefault(child, []).append(depth + 1)
                pending.append((child, depth + 1))
        ancestry.extend(
            (source, target, min(depths), max(depths)) for target, depths in distances.items()
        )
    write_table(
        vocab / "CONCEPT_ANCESTOR.csv",
        [
            "ancestor_concept_id",
            "descendant_concept_id",
            "min_levels_of_separation",
            "max_levels_of_separation",
        ],
        ancestry,
        separator="\t",
        seed=seed,
    )
    write_table(
        vocab / "VOCABULARY.csv",
        ["vocabulary_id", "vocabulary_version"],
        [
            ("SNOMED", "synthetic-c0014-v1"),
            ("ICD9CM", "synthetic-c0014-v1"),
            ("ICD10CM", "synthetic-c0014-v1"),
        ],
        separator="\t",
        seed=seed,
    )
    patients = [] if empty else ["1", "2", "3"] + (["4"] if count_one else [])
    visits = (
        []
        if empty
        else [
            ("1", "11", "2020-01-01 00:00:00"),
            ("1", "12", "2020-01-02 00:00:00"),
            ("2", "21", "2020-01-01 00:00:00"),
            ("3", "31", "2020-01-01 00:00:00"),
        ]
    )
    if count_one and not empty:
        visits.append(("4", "41", "2020-01-01 00:00:00"))
    diagnoses = (
        []
        if empty
        else [
            ("1", "11", "1", "0010", "9"),
            ("1", "12", "1", "0010", "10"),
            ("2", "21", "1", "I214", "10"),
            ("2", "21", "1", "k.1", "10"),
            ("3", "31", "1", "k.1", "10"),
            ("3", "31", "1", "k.1", "10"),
        ]
    )
    if count_one and not empty:
        diagnoses += [("4", "41", "1", "Z1", "10"), ("4", "41", "2", "Z1", "10")]
    mimic = destination / "mimic4"
    write_table(mimic / "hosp/patients.csv.gz", ["subject_id"], [(p,) for p in patients], seed=seed)
    write_table(
        mimic / "hosp/admissions.csv.gz", ["subject_id", "hadm_id", "admittime"], visits, seed=seed
    )
    write_table(mimic / "icu/icustays.csv.gz", ["subject_id", "intime"], [], seed=seed)
    write_table(
        mimic / "hosp/diagnoses_icd.csv.gz",
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        diagnoses,
        seed=seed,
    )
    omop = destination / "omop"
    write_table(
        omop / "person.csv",
        ["person_id", "year_of_birth", "month_of_birth", "day_of_birth"],
        [(p, 1980, 1, 1) for p in patients],
        seed=seed,
    )
    write_table(
        omop / "visit_occurrence.csv",
        ["person_id", "visit_occurrence_id", "visit_start_datetime", "visit_end_datetime"],
        [(p, v, date, date) for p, v, date in visits],
        seed=seed,
    )
    write_table(omop / "death.csv", ["person_id", "death_datetime"], [], seed=seed)
    admission_dates = {v: date for _, v, date in visits}
    identities = {row: str(i + 1) for i, row in enumerate(dict.fromkeys(diagnoses))}
    write_table(
        omop / "condition_occurrence.csv",
        [
            "person_id",
            "condition_occurrence_id",
            "condition_concept_id",
            "condition_start_datetime",
            "visit_occurrence_id",
            "condition_source_value",
        ],
        [
            (
                p,
                identities[(p, visit, seq, code, version)],
                0,
                admission_dates[visit],
                visit,
                f"ICD{'9CM' if version == '9' else '10CM'}:{code}",
            )
            for p, visit, seq, code, version in diagnoses
        ],
        seed=seed,
    )
    write_table(
        destination / "cohort.csv",
        ["patient_id", "label", "index_date"],
        [(p, int(p == "1"), "2020-01-02" if p == "1" else "") for p in patients],
        seed=seed,
    )
    return destination
