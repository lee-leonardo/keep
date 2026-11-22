"""Matched-input comparison with untouched, pinned author history functions."""

import contextlib
import hashlib
import io
import json
import subprocess
from pathlib import Path

from . import KEEP_BASE
from .bundle import read_jsonl
from .projection import project


def compare_author_histories(bundle):
    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    cohort = {
        p: (spec["label"], spec["index_date"]) for p, spec in manifest["projection_cohort"].items()
    }
    records = list(read_jsonl(Path(bundle) / "histories.jsonl"))
    return compare_author_projection(records, cohort)


def compare_author_projection(records, cohort):
    """Run both implementations on the same supported, date-only input."""
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", f"{KEEP_BASE}:cohorts_patients/create_cohort_sentence.py"], cwd=root
    )
    namespace = {"__name__": "pinned_author_history"}
    exec(compile(source, "pinned_author_history.py", "exec"), namespace)
    flattened = []
    for row in records:
        if row["rolled_up_concepts"]:
            if row["event_time"] is None or not row["event_time"].endswith("T00:00:00.000000"):
                raise ValueError("Author comparison uses the shared date-only input domain")
            year, month, day = row["event_time"][:10].split("-")
            flattened.extend(
                (row["patient_id"], code, f"{day}/{month}/{year}")
                for code in row["rolled_up_concepts"]
            )
    frame = pd.DataFrame(flattened, columns=["eid", "concept_id", "condition_start_date"])
    positive = {p: date[:10] for p, (label, date) in cohort.items() if label == 1}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cases = namespace["process_positive_cases"](frame, positive)
        controls = namespace["process_negative_cases"](frame, set(positive), list(cohort))
    actual = {
        p: {"label": cohort[p][0], "concepts": sorted(set(codes), key=int)}
        for p, codes in (controls | cases).items()
    }
    candidate = {
        row["patient_id"]: {"label": row["label"], "concepts": sorted(row["concept_ids"], key=int)}
        for row in project(records, cohort)
    }
    if candidate != actual:
        raise AssertionError(
            "History membership, labels or concept sets differ from author functions"
        )
    return {
        "status": "equal",
        "reference_revision": KEEP_BASE,
        "reference_file": "cohorts_patients/create_cohort_sentence.py",
        "reference_source_sha256": hashlib.sha256(source).hexdigest(),
        "shared_domain": "already_rolled_up_unique_diagnosis_to_target_rows_date_only",
        "positive_policy": "strictly_before_index_date",
        "control_policy": "uncensored",
        "included_patients": len(actual),
        "included_positive_patients": sum(row["label"] == 1 for row in actual.values()),
        "included_negative_patients": sum(row["label"] == 0 for row in actual.values()),
        "patient_membership": "exact",
        "concept_sets": "exact",
        "included_labels": "exact",
        "ordering": "candidate_contract_checked_separately",
    }
