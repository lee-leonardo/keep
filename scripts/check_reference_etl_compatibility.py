"""Recheck the ETL feedback using generated inputs and read-only sibling code.

All generated configs, CSVs, caches, logs and outputs go below --output. No
existing ETL config, private input, notebook, source file or result is modified.
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def run(command, *, cwd, env, log):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=240)
    log.write_text(result.stdout + result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etl-checkout", type=Path, required=True)
    parser.add_argument("--etl-python", type=Path, required=True)
    parser.add_argument("--pyhealth-checkout", type=Path, required=True)
    parser.add_argument("--pyhealth-revision", default="46261db82f68ecd5820c204f333f288a6720ce13")
    parser.add_argument("--keep-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    etl = args.etl_checkout.absolute()
    # Preserve virtual-environment Python symlinks rather than resolving them.
    etl_python, keep_python = str(args.etl_python.absolute()), str(args.keep_python.absolute())
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYHEALTH_REQUIRE_CACHE_DIR": "1",
            "OMP_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "2",
            "MPLCONFIGDIR": str(output / "matplotlib"),
            "XDG_CACHE_HOME": str(output / "user-cache"),
        }
    )
    summary = {
        "input_kind": "generated_synthetic_only",
        "cases": {},
        "claim": "Probe compatibility only; no author-artifact or published-result claim",
    }
    for name in ("baseline", "retired_icd_source", "noncondition_and_invalid_targets"):
        case = output / name
        case.mkdir()
        inputs = case / "inputs"
        generation = run(
            [
                etl_python,
                "-B",
                "-m",
                "keep_comparison",
                "--generate-fixture",
                "--output",
                str(inputs),
            ],
            cwd=etl,
            env=env,
            log=case / "generation.log",
        )
        if generation.returncode:
            summary["cases"][name] = {"status": "error", "stage": "fixture_generation"}
            continue
        if name == "retired_icd_source":
            path = inputs / "CONCEPT.csv"
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                columns, rows = reader.fieldnames, list(reader)
            for row in rows:
                if row["concept_id"] == "11":
                    row["invalid_reason"] = "D"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=columns, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
        elif name == "noncondition_and_invalid_targets":
            with (inputs / "CONCEPT_RELATIONSHIP.csv").open("a") as stream:
                stream.write("11\t902\tMaps to\t\n11\t903\tMaps to\t\n")
        config = case / "comparison.synthetic.toml"
        # JSON-quoted strings are valid TOML basic strings for these local paths.
        sections = {
            "inputs": {"directory": str(inputs)},
            "pyhealth": {
                "checkout": str(args.pyhealth_checkout.absolute()),
                "revision": args.pyhealth_revision,
            },
            "keep": {"checkout": str(root), "python": keep_python, "revision": revision},
        }
        config.write_text(
            'profile="baseline"\n'
            + "".join(
                "\n["
                + section
                + "]\n"
                + "".join(key + "=" + json.dumps(value) + "\n" for key, value in values.items())
                for section, values in sections.items()
            )
        )
        config.chmod(0o600)
        result = run(
            [
                etl_python,
                "-B",
                "-m",
                "keep_comparison",
                "--config",
                str(config),
                "--output",
                str(case / "run"),
            ],
            cwd=etl,
            env=env,
            log=case / "execution.log",
        )
        report_path = case / "run/summary.json"
        if not report_path.is_file():
            summary["cases"][name] = {
                "status": "error",
                "stage": "comparison_execution",
                "exit_code": result.returncode,
            }
            continue
        report = json.loads(report_path.read_text())
        statuses = {boundary: value["status"] for boundary, value in report["boundaries"].items()}
        record = {
            "exit_code": result.returncode,
            "boundaries": statuses,
            "input_sha256": report.get("inputs"),
            "harness_source": report.get("harness_source"),
            "runtime": report.get("runtime"),
            "author_reference_coverage": report["author_reference_coverage"],
        }
        native_path = case / "run/keep/manifest.json"
        if native_path.is_file():
            manifest = json.loads(native_path.read_text())
            record.update(
                {
                    "keep_source": manifest["implementation"],
                    "mapping_contract": manifest["mapping_contract"],
                    "timestamp_contract": manifest["timestamp_contract"],
                }
            )
        record["status"] = (
            "equal"
            if result.returncode == 0 and len(statuses) == 4 and set(statuses.values()) == {"equal"}
            else "failed"
        )
        summary["cases"][name] = record
    values = [
        "2020-01-01T00:00:00.1234561",
        "2020-01-01T00:00:00.1234569",
        "2020-01-01T00:00:00.123456",
    ]
    code = """import json, sys
from keep_reference.io import event_time
rows=[]
for value in json.loads(sys.argv[1]):
 try: rows.append({"status":"accepted","value":event_time(value)})
 except ValueError: rows.append({"status":"rejected"})
print(json.dumps(rows))
"""
    timestamps = run(
        [keep_python, "-S", "-c", code, json.dumps(values)],
        cwd=root,
        env=env,
        log=output / "timestamp.log",
    )
    summary["timestamps"] = (
        json.loads(timestamps.stdout) if timestamps.returncode == 0 else {"status": "error"}
    )
    summary["passed"] = all(
        case["status"] == "equal" for case in summary["cases"].values()
    ) and summary["timestamps"] == [
        {"status": "rejected"},
        {"status": "rejected"},
        {"status": "accepted", "value": values[2]},
    ]
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "cases": {name: record["status"] for name, record in summary["cases"].items()},
                "timestamp_precision": "passed" if summary["passed"] else "inspect_summary",
            },
            sort_keys=True,
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
