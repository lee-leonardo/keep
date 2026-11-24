"""Strict table I/O and canonical serialization, independent of domain algorithms."""

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def json_text(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def fingerprint(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_csv(path, required, *, athena=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(
            stream,
            delimiter="\t" if athena else ",",
            quoting=csv.QUOTE_NONE if athena else csv.QUOTE_MINIMAL,
            strict=True,
        )
        if (
            not reader.fieldnames
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or not set(required) <= set(reader.fieldnames)
        ):
            raise ValueError("Input table has missing or repeated column names")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Input table has malformed row width")
            yield row


def concept_id(value, *, nullable=False, allow_zero=False):
    if nullable and value in {None, "", "0"}:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value.strip()):
        raise ValueError("Concept IDs require decimal integers")
    number = int(value)
    if nullable and number == 0:
        return None
    if number == 0 and not allow_zero:
        raise ValueError("Concept IDs must be positive")
    return str(number)


def opaque_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty source identity is required")
    return value


def event_time(datetime_value, date_value=""):
    value = datetime_value or date_value
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("Source event time must be an ISO string")
    # datetime.fromisoformat silently discards digits after the sixth. Reject
    # textual precision outside the contract before distinct events can merge.
    if any(len(part) > 6 for part in re.findall(r"[.,]([0-9]+)", value)):
        raise ValueError("Source timestamp precision exceeds supported microseconds")
    try:
        time = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("Invalid ISO source event time") from None
    if time.tzinfo is not None:
        raise ValueError("The common profile requires timezone-naive event times")
    return time.isoformat(timespec="microseconds")


def write_json(path, value):
    Path(path).write_text(json_text(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(json_text(row) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_tsv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path, columns):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames != columns:
            raise ValueError("Comparison table columns do not match the schema")
        rows = list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError("Malformed comparison table")
        return rows
