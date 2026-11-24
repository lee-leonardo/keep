"""Prediction histories are separate from complete-history empirical counts."""

from collections import defaultdict
from datetime import datetime


def timestamp(value):
    if value is None or value == "":
        return None
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.tzinfo is not None:
        raise ValueError("Use timezone-naive source times consistently")
    return result.isoformat(timespec="microseconds")


def project(records, cohort):
    """Threshold after strict positive-case censoring; controls are uncensored.

    Order concepts by second qualifying occurrence time, then numeric concept ID.
    Patient and concept membership, rather than this order, is the author-code
    comparator. Missing positive-case event times are errors.
    """
    events = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in records:
        key = (row["patient_id"], row["diagnosis_id"])
        if key in seen:
            raise ValueError("Projection requires unique source diagnoses")
        seen.add(key)
        patient = row["patient_id"]
        if patient not in cohort:
            continue
        label, cutoff = cohort[patient]
        time = row["event_time"]
        if label == 1:
            if cutoff is None:
                raise ValueError("Positive cases require an explicit index date")
            if row["rolled_up_concepts"] and time is None:
                raise ValueError("Positive-case projection requires event timestamps")
            if time is None or time >= cutoff:
                continue
        for code in row["rolled_up_concepts"]:
            events[patient][code].append(time or "")
    projected = []
    for patient in sorted(cohort):
        eligible = [
            (sorted(times)[1], int(code), code)
            for code, times in events[patient].items()
            if len(times) >= 2
        ]
        if eligible:
            projected.append(
                {
                    "patient_id": patient,
                    "label": cohort[patient][0],
                    "concept_ids": [code for _, _, code in sorted(eligible)],
                }
            )
    return projected
