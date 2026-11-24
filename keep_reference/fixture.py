"""Common generated CSVs. Sharing input generation does not share producers."""

import csv
from pathlib import Path


def generate_fixture(output, *, seed=None):
    # Only input generation is shared with the historical synthetic example.
    # No reader/mapper/graph/roll-up/counter is imported or reused by the build.
    from keep_preprocessing.fixtures import generate, write_table

    root = generate(output, seed=seed, count_one=True)

    def extend(path, extra, *, separator=","):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=separator)
            columns, rows = reader.fieldnames, list(reader)
        rows.extend(extra)
        return columns, rows

    def save(path, columns, rows, *, separator=","):
        write_table(
            path,
            columns,
            ([row.get(c, "") for c in columns] for row in rows),
            separator=separator,
            seed=seed,
        )

    path = root / "omop/person.csv"
    columns, rows = extend(
        path,
        [
            {"person_id": p, "year_of_birth": "1980", "month_of_birth": "1", "day_of_birth": "1"}
            for p in ["5", "6"]
        ],
    )
    save(path, columns, rows)
    path = root / "omop/condition_occurrence.csv"
    extra = []
    for number, concept, source in [
        (601, "300", "BAD:ignored"),
        (602, "600", ""),
        (603, "700", ""),
        (604, "800", ""),
        (605, "999999", ""),
        (606, "0", "ICD10CM:NOMAP"),
        (607, "0", ""),
        (608, "0", "BAD:0010"),
    ]:
        extra.append(
            {
                "person_id": "6",
                "condition_occurrence_id": str(number),
                "condition_concept_id": concept,
                "condition_source_value": source,
                "visit_occurrence_id": "",
                "condition_start_datetime": "2020-01-03 01:02:03.123456" if number == 601 else "",
                "condition_start_date": "1999-01-01"
                if number == 601
                else ("2020-01-03" if number == 602 else ""),
            }
        )
    columns, rows = extend(path, extra)
    columns.append("condition_start_date")
    save(path, columns, rows)
    path = root / "vocabulary/CONCEPT.csv"
    columns, rows = extend(
        path,
        [
            {
                "concept_id": "250",
                "concept_name": "Unused retained W",
                "domain_id": "Condition",
                "vocabulary_id": "SNOMED",
                "concept_class_id": "Clinical Finding",
                "standard_concept": "S",
                "concept_code": "250",
                "invalid_reason": "",
            },
            {
                "concept_id": "990",
                "concept_name": "Closure-only deep Y",
                "domain_id": "Condition",
                "vocabulary_id": "SNOMED",
                "concept_class_id": "Clinical Finding",
                "standard_concept": "S",
                "concept_code": "990",
                "invalid_reason": "",
            },
        ],
        separator="\t",
    )
    save(path, columns, rows, separator="\t")
    path = root / "vocabulary/CONCEPT_ANCESTOR.csv"
    extra = [
        {
            "ancestor_concept_id": str(a),
            "descendant_concept_id": "990",
            "min_levels_of_separation": str(n),
            "max_levels_of_separation": str(n),
        }
        for a, n in [(4274025, 6), (10, 5), (20, 4), (30, 3), (40, 2), (200, 1), (990, 0)]
    ]
    extra += [
        {
            "ancestor_concept_id": str(a),
            "descendant_concept_id": "250",
            "min_levels_of_separation": str(n),
            "max_levels_of_separation": str(n),
        }
        for a, n in [(4274025, 5), (10, 4), (20, 3), (30, 2), (40, 1), (250, 0)]
    ]
    columns, rows = extend(path, extra, separator="\t")
    save(path, columns, rows, separator="\t")
    config = """schema_version = "keep-reference-config-v1"
vocabulary_release = "synthetic-c0014-v1"
fixture_version = "keep-reference-common-v1"

[inputs]
person = "omop/person.csv"
visit_occurrence = "omop/visit_occurrence.csv"
death = "omop/death.csv"
condition_occurrence = "omop/condition_occurrence.csv"
CONCEPT = "vocabulary/CONCEPT.csv"
CONCEPT_RELATIONSHIP = "vocabulary/CONCEPT_RELATIONSHIP.csv"
CONCEPT_ANCESTOR = "vocabulary/CONCEPT_ANCESTOR.csv"
VOCABULARY = "vocabulary/VOCABULARY.csv"

[settings]
max_depth = 5
target_policy = "all_retained"
path_source = "closure"
"""
    (Path(root) / "reference.toml").write_text(config, encoding="utf-8")
    return root
