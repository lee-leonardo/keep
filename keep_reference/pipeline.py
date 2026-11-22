"""Public independent stage composition and atomic comparison-bundle build."""

import hashlib
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import KEEP_BASE, SCHEMA_VERSION
from .bundle import load_bundle, write_manifest, write_payloads
from .config import (
    MAPPING_CONTRACT,
    TIMESTAMP_CONTRACT,
    ReferenceConfig,
    checked_settings,
    load_config,
)
from .counts import count_cooccurrence
from .histories import read_histories
from .io import file_digest, fingerprint
from .ontology import load_ontology
from .rollup import rollup_histories, rollup_sources, source_universe


@dataclass
class ReferenceResult:
    patient_ids: tuple
    ontology: object
    histories: list
    source_ids: tuple
    observed_source_ids: tuple
    rollup_results: dict
    rolled_histories: list
    patient_presence: list
    counts: list


def compute(config):
    """Return all native boundary structures without writing or training anything."""
    if not isinstance(config, ReferenceConfig):
        config = load_config(config)
    checked_settings(config.settings)
    ontology = load_ontology(config.inputs, config.settings)
    patients, histories = read_histories(config.inputs, ontology, config.patient_ids)
    sources, observed = source_universe(ontology, histories, config.source_ids)
    targets = rollup_sources(ontology, sources, config.settings)
    rolled = rollup_histories(histories, targets)
    presence, counts = count_cooccurrence(rolled, patients, ontology.retained_ids)
    return ReferenceResult(
        patients, ontology, histories, sources, observed, targets, rolled, presence, counts
    )


def source_identity(expected_head=None, expected_checkout=None):
    package = Path(__file__).resolve().parent
    checkout = package.parent
    if expected_checkout is not None and Path(expected_checkout).resolve() != checkout:
        raise ValueError("Reference imported from a different KEEP checkout")

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "--no-optional-locks", *args], cwd=checkout, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            raise ValueError("Unable to verify the declared KEEP checkout") from None

    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != checkout:
        raise ValueError("Reference imports must resolve inside the supplied KEEP repository")
    head = git("rev-parse", "HEAD").decode().strip()
    if expected_head is not None and head != expected_head:
        raise ValueError("KEEP checkout revision does not match the selected configuration")
    sources = {"keep_reference/" + p.name: file_digest(p) for p in sorted(package.glob("*.py"))}
    executed = {}
    for name, module in list(sys.modules.items()):
        if name == "keep_reference" or name.startswith("keep_reference."):
            location = getattr(module, "__file__", None)
            if location is None or Path(location).resolve().parent != package:
                raise ValueError("Mixed reference module origins are not supported")
            executed["keep_reference/" + Path(location).name] = file_digest(location)
    return {
        "released_base": KEEP_BASE,
        "checkout_head": head,
        "module": "keep_reference",
        "source_files": sources,
        "executed_source_files": dict(sorted(executed.items())),
        "source_sha256": fingerprint(sources),
        "relevant_diff_sha256": hashlib.sha256(
            git("diff", "--no-ext-diff", "HEAD", "--", "keep_reference")
        ).hexdigest(),
        "untracked_source_files": sorted(
            git("ls-files", "--others", "--exclude-standard", "--", "keep_reference")
            .decode()
            .splitlines()
        ),
    }


def build(config, output, *, expected_checkout=None):
    """Write keep-comparison-v1 from explicit CSVs; existing outputs are refused."""
    if not isinstance(config, ReferenceConfig):
        config = load_config(config)
    output = Path(output)
    if output.exists():
        raise ValueError("Select a fresh reference output directory")
    identity = source_identity(config.expected_keep_head, expected_checkout)
    inputs = {name: file_digest(path) for name, path in sorted(config.inputs.items())}
    result = compute(config)
    ontology = result.ontology

    def nodes(codes):
        return [
            {
                "row_id": i,
                "concept_id": code,
                "label": ontology.concepts[code]["concept_name"],
                "root_depth": ontology.depths[code],
            }
            for i, code in enumerate(codes)
        ]

    def edges(pairs):
        return [{"parent_id": p, "child_id": c, "relation": "is_a"} for p, c in pairs]

    retained = set(ontology.retained_ids)
    tables = {
        "concepts.tsv": nodes(ontology.retained_ids),
        "uncut_concepts.tsv": nodes(sorted(ontology.depths, key=int)),
        "edges.tsv": edges(
            (p, c) for p, c in ontology.direct_edges if p in retained and c in retained
        ),
        "uncut_edges.tsv": edges(ontology.direct_edges),
        "closure_links.tsv": [
            {"ancestor_id": a, "descendant_id": d, "distance": n}
            for a, d, n in ontology.closure_links
        ],
        "rollup.tsv": [
            {"source_id": source, "target_id": target}
            for source, value in result.rollup_results.items()
            for target in value["target_ids"]
        ],
        "rollup_status.tsv": [
            {
                "source_id": source,
                "root_depth": "" if value["root_depth"] is None else value["root_depth"],
                "reason": value["reason"],
            }
            for source, value in result.rollup_results.items()
        ],
        "counts.tsv": result.counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".keep-reference-", dir=output.parent))
    try:
        artifacts = write_payloads(
            stage, tables, result.histories, result.rolled_histories, result.patient_presence
        )
        declarations = {
            "histories": ("keep_reference.histories.read_histories", ["histories.jsonl"], []),
            "ontology": (
                "keep_reference.ontology.load_ontology",
                [
                    "concepts.tsv",
                    "edges.tsv",
                    "uncut_concepts.tsv",
                    "uncut_edges.tsv",
                    "closure_links.tsv",
                ],
                [],
            ),
            "rollup": (
                "keep_reference.rollup.rollup_sources",
                ["rollup.tsv", "rollup_status.tsv", "rolled_histories.jsonl"],
                ["ontology", "histories"],
            ),
            "counts": (
                "keep_reference.counts.count_cooccurrence",
                ["counts.tsv", "patient_presence.jsonl"],
                ["histories", "rollup", "ontology"],
            ),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "side": "keep",
            "settings": config.settings,
            "mapping_contract": dict(MAPPING_CONTRACT),
            "timestamp_contract": dict(TIMESTAMP_CONTRACT),
            "patient_ids": list(result.patient_ids),
            "row_concept_ids": list(ontology.retained_ids),
            "column_concept_ids": list(ontology.retained_ids),
            "matrix_shape": [len(retained), len(retained)],
            "source_concept_ids": list(result.source_ids),
            "observed_source_concept_ids": list(result.observed_source_ids),
            "inputs": inputs,
            "vocabulary_release": config.vocabulary_release,
            "fixture_version": config.fixture_version,
            "cohort_selection": None if config.patient_ids is None else sorted(config.patient_ids),
            "implementation": identity,
            "environment": {
                "python": platform.python_version(),
                "external_packages": {},
                "dependency_lock_sha256": None,
                "dependency_profile": "python_standard_library_only",
            },
            "native_relation_mapping": {"Is a": "is_a"},
            "native_order": {
                "concepts": "numeric_id",
                "histories": "patient_then_diagnosis_opaque_id",
            },
            "status_mapping": {
                r: r
                for r in [
                    "retained",
                    "rolled_up",
                    "unknown_code",
                    "outside_domain",
                    "unreachable",
                    "no_target",
                ]
            },
            "boundaries": {
                name: {
                    "producer_entry_point": producer,
                    "provenance_kind": "keep_reconstruction",
                    "execution_status": "produced",
                    "executed_source_sha256": identity["source_sha256"],
                    "executed_source_files": identity["executed_source_files"],
                    "input_sha256": inputs,
                    "output_sha256": {file: artifacts[file] for file in files},
                    "upstream_boundaries": upstream,
                    "supplied_common_inputs": [],
                }
                for name, (producer, files, upstream) in declarations.items()
            },
            "author_reference_coverage": {name: "missing_reference" for name in declarations},
            "artifacts": artifacts,
        }
        write_manifest(stage, manifest)
        load_bundle(stage)
        if (
            any(file_digest(config.inputs[name]) != value for name, value in inputs.items())
            or source_identity(config.expected_keep_head, expected_checkout) != identity
        ):
            raise ValueError("Input or executed reference source changed during the build")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return load_bundle(output)
