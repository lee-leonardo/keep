"""Public CPU preparation and bundle validation commands."""

import argparse
import json

from .bundle import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="Generate public synthetic inputs")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--shuffle-seed", type=int)
    fixture.add_argument("--count-one", action="store_true")
    fixture.add_argument("--empty", action="store_true")
    build_parser = commands.add_parser("build", help="Build a fresh versioned input bundle")
    build_parser.add_argument("--input-format", choices=["omop", "mimic4"], required=True)
    for name in ["source", "vocabulary", "vocabulary-release", "output", "cache-dir"]:
        build_parser.add_argument("--" + name, required=True)
    build_parser.add_argument("--cohort", help="Optional CSV: patient_id,label,index_date")
    build_parser.add_argument(
        "--fit-patients", help="Optional newline-separated fitting patient IDs"
    )
    build_parser.add_argument(
        "--target-policy", choices=["all_retained", "first_retained"], default="all_retained"
    )
    build_parser.add_argument("--path-source", choices=["direct", "closure"], default="closure")
    build_parser.add_argument("--max-depth", type=int, default=5)
    validate_parser = commands.add_parser(
        "validate", help="Validate bundle integrity and semantics"
    )
    validate_parser.add_argument("bundle")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "fixture":
        from .fixtures import generate

        args["seed"] = args.pop("shuffle_seed")
        args["destination"] = args.pop("output")
        result = {"fixture": str(generate(**args))}
    elif command == "build":
        from .build import build

        result = build(**args)
    else:
        result = validate(args["bundle"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
