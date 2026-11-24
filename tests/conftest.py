import os
from pathlib import Path

# Configure library caches before PyHealth or plotting dependencies import.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".keep-work" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".keep-work" / "user-cache"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYHEALTH_REQUIRE_CACHE_DIR"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "2"
