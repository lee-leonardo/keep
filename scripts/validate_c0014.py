"""Run the synthetic checks and write a sanitized, reproducible evidence summary."""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-python", required=True)
    parser.add_argument(
        "--output", required=True, help="Fresh local directory for logs and evidence"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence.json"
    junit = output / "tests.xml"
    env = {
        **os.environ,
        # Preserve the venv entry point instead of resolving to the base Python.
        "KEEP_TRAINING_PYTHON": str(Path(args.training_python).absolute()),
        "KEEP_EVIDENCE_OUTPUT": str(evidence),
        "MPLCONFIGDIR": str(output / "matplotlib"),
        "XDG_CACHE_HOME": str(output / "cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (output / "tests.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        print((output / "tests.log").read_text())
        raise SystemExit(result.returncode)
    summary = json.loads(evidence.read_text())
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    summary["test_suite"] = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ["tests", "failures", "errors", "skipped"]
    }
    summary["command"] = (
        ".venv-preprocessing/bin/python scripts/validate_c0014.py --training-python .venv/bin/python --output LOCAL_REVIEW_DIRECTORY"
    )
    # Deliberately omit log contents, temporary directories, hostnames and paths.
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"passed": summary["test_suite"]["tests"], "summary": str(output / "summary.json")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
