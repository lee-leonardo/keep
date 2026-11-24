"""Deterministic KEEP input bundles. Importing this package does no I/O."""

SCHEMA_VERSION = 1
KEEP_BASE = "89cb4823212d1a89cd57bb717731b68e7f10c789"
PYHEALTH_REVISION = "46261db82f68ecd5820c204f333f288a6720ce13"
# Baseline semantics fixed by schema v1. The builder records these and the
# validator rejects any bundle that disagrees; import them instead of
# redefining, so the two can never drift apart.
FIXED_PARAMETERS = {
    "root": "4274025",
    "depth_measure": "min",
    "include_root": False,
    "orphan_rescue": False,
    "presence_threshold": 2,
    "threshold_stage": "post_rollup",
    "diagonal": "patient_prevalence",
    "history_window": "complete",
    "count_unit": "patient",
    "mimic4_time": "admission",
}
