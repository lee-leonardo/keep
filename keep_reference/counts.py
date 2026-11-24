"""Independent complete-history presence sets and integer patient-pair counts."""

from collections import Counter

from .io import concept_id, opaque_id


def count_cooccurrence(rolled_histories, patient_ids, row_concept_ids, column_concept_ids=None):
    """Also supports conditional stage experiments with explicitly supplied inputs.

    Patient/diagnosis identity survives deduplication. Both native axis maps are
    explicit; no training filters, weighting or temporal censoring are applied.
    """
    patients = list(patient_ids)
    rows = list(row_concept_ids)
    columns = rows if column_concept_ids is None else list(column_concept_ids)
    for patient in patients:
        opaque_id(patient)
    if any(concept_id(code) != code for code in rows + columns):
        raise ValueError("Matrix axes require canonical decimal concept strings")
    if (
        len(set(patients)) != len(patients)
        or len(set(rows)) != len(rows)
        or len(set(columns)) != len(columns)
        or set(rows) != set(columns)
    ):
        raise ValueError("Patient and matrix axis identities must be unique with full coverage")
    row_index, column_index = (
        {code: i for i, code in enumerate(rows)},
        {code: i for i, code in enumerate(columns)},
    )
    frequencies = {patient: Counter() for patient in patients}
    diagnoses = {}
    for record in rolled_histories:
        key = (opaque_id(record["patient_id"]), opaque_id(record["diagnosis_id"]))
        if key in diagnoses:
            if diagnoses[key] != record:
                raise ValueError("Conflicting copies in conditional rolled-up input")
            continue
        diagnoses[key] = record
        if key[0] not in frequencies:
            raise ValueError("Rolled-up input contains an undeclared patient")
        targets = record["rolled_up_concepts"]
        if not set(targets) <= row_index.keys():
            raise ValueError("Rolled-up targets fall outside the declared full axes")
        frequencies[key[0]].update(set(targets))
    presence = [
        {
            "patient_id": p,
            "concept_ids": sorted((code for code, n in frequencies[p].items() if n >= 2), key=int),
        }
        for p in sorted(patients)
    ]
    counts = Counter()
    for row in presence:
        for first in row["concept_ids"]:
            for second in row["concept_ids"]:
                counts[(row_index[first], column_index[second])] += 1
    triples = [{"row": i, "column": j, "count": n} for (i, j), n in sorted(counts.items())]
    return presence, triples
