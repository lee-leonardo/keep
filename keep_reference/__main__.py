"""Independent raw-CSV reference stages for an external ETL comparison harness."""

import argparse
import csv
import json
import sys
from pathlib import Path

from . import SCHEMA_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser(
        "fixture", help="Generate public common CSVs and an example config"
    )
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--shuffle-seed", type=int)
    builder = commands.add_parser("build", help="Build a fresh independent comparison bundle")
    builder.add_argument("--config", required=True)
    builder.add_argument("--output", required=True)
    validator = commands.add_parser(
        "validate", help="Check internal integrity; does not assert parity"
    )
    validator.add_argument("bundle")
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            from .fixture import generate_fixture

            generate_fixture(args.output, seed=args.shuffle_seed)
            result = {"status": "generated", "fixture_version": "keep-reference-common-v1"}
        elif args.command == "build":
            from .pipeline import build

            build(args.config, args.output, expected_checkout=Path.cwd())
            result = {
                "schema_version": SCHEMA_VERSION,
                "side": "keep",
                "status": "built",
                "provenance_kind": "keep_reconstruction",
            }
        else:
            from .bundle import validate_bundle

            result = validate_bundle(args.bundle)
    except (ValueError, KeyError, TypeError, csv.Error) as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": "Reference input or bundle contract validation failed",
                }
            ),
            file=sys.stderr,
        )
        return 1
    except OSError:
        # Avoid logging an explicitly supplied private path or source row.
        print(json.dumps({"status": "error", "error_type": "InputOutputError"}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
