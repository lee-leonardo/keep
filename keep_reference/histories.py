"""Read raw OMOP CSV histories, preserving source identity before roll-up."""

from .io import concept_id, event_time, opaque_id, read_csv


def read_histories(inputs, ontology, patient_ids=None):
    people = {}
    for row in read_csv(inputs["person"], ["person_id"]):
        patient = opaque_id(row["person_id"])
        if people.setdefault(patient, row) != row:
            raise ValueError("Conflicting person rows")
    selected = set(people) if patient_ids is None else set(patient_ids)
    if not selected <= people.keys():
        raise ValueError("Cohort selection contains an unknown patient")
    visits = {}
    for row in read_csv(inputs["visit_occurrence"], ["person_id", "visit_occurrence_id"]):
        patient, visit = opaque_id(row["person_id"]), opaque_id(row["visit_occurrence_id"])
        if patient not in people:
            raise ValueError("Visit references an unknown patient")
        if visits.setdefault((patient, visit), row) != row:
            raise ValueError("Conflicting visit rows")
    for row in read_csv(inputs["death"], ["person_id"]):
        if opaque_id(row["person_id"]) not in people:
            raise ValueError("Death row references an unknown patient")
    diagnoses = {}
    source_copies = {}
    for row in read_csv(
        inputs["condition_occurrence"],
        [
            "person_id",
            "condition_occurrence_id",
            "condition_concept_id",
            "condition_source_value",
            "visit_occurrence_id",
        ],
    ):
        patient, identity = opaque_id(row["person_id"]), opaque_id(row["condition_occurrence_id"])
        if patient not in people:
            raise ValueError("Diagnosis references an unknown patient")
        if not {"condition_start_datetime", "condition_start_date"} & row.keys():
            raise ValueError("A condition start datetime or date column is required")
        # Validate duplicate copies before projection, mapping or canonicalization.
        key = (patient, identity)
        if source_copies.setdefault(key, row) != row:
            raise ValueError("Conflicting source diagnosis copies")
        source_concept = concept_id(row["condition_concept_id"], nullable=True)
        original = row["condition_source_value"] or None
        if source_concept is not None:
            mapped, reason = [source_concept], None
        elif original is None or not original.strip():
            mapped, reason = [], "missing_source"
        else:
            vocabulary, colon, code = original.strip().partition(":")
            if not colon or vocabulary not in {"ICD9CM", "ICD10CM"} or not code.strip():
                mapped, reason = [], "unsupported_source_format"
            else:
                mapped = ontology.map_icd(vocabulary, code)
                reason = None if mapped else "empty_icd_mapping"
        record = {
            "patient_id": patient,
            "diagnosis_id": identity,
            "visit_id": row["visit_occurrence_id"] or None,
            "event_time": event_time(
                row.get("condition_start_datetime", ""), row.get("condition_start_date", "")
            ),
            "source_value": original,
            "source_concept_id": source_concept,
            "mapped_concepts": mapped,
            "unmapped_reason": reason,
        }
        if patient in selected:
            diagnoses[key] = record
    return tuple(sorted(selected)), [diagnoses[key] for key in sorted(diagnoses)]
