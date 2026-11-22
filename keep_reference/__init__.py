"""Independent CSV reconstruction for KEEP comparison, with no third-party imports.

This is fork-local reconstruction code, not released author preprocessing.
"""

SCHEMA_VERSION = "keep-comparison-v1"
KEEP_BASE = "89cb4823212d1a89cd57bb717731b68e7f10c789"


def __getattr__(name):
    if name in {"build", "compute", "load_config"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name in {"load_bundle", "validate_bundle"}:
        from . import bundle

        return getattr(bundle, name)
    raise AttributeError(name)
