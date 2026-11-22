"""Check the pinned public PyHealth package and record executable source identity."""

import importlib.metadata
import json
from pathlib import Path

from . import PYHEALTH_REVISION
from .bundle import digest, object_digest


def check_runtime():
    import pyhealth
    from pyhealth.tasks import KeepConditionHistoryTask

    distribution = importlib.metadata.distribution("pyhealth")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != PYHEALTH_REVISION:
        raise ValueError(
            "Install the exact PyHealth VCS revision in requirements-preprocessing.lock"
        )
    if KeepConditionHistoryTask.task_name != "keep_condition_history_v2":
        raise ValueError("The public v2 diagnosis history task is required")
    root = Path(pyhealth.__file__).parent
    sources = {
        name: digest(root / name)
        for name in [
            "tasks/keep.py",
            "processors/ancestor_rollup_processor.py",
            "graph/cooccurrence.py",
            "medcode/omop/vocabulary.py",
            "medcode/omop/icd_mapper.py",
            "datasets/base_dataset.py",
            "datasets/sample_dataset.py",
        ]
    }
    return {"revision": PYHEALTH_REVISION, "source_sha256": object_digest(sources)}, {
        name: importlib.metadata.version(name)
        for name in ["pyhealth", "torch", "polars", "pandas", "numpy", "litdata", "pyarrow"]
    }
