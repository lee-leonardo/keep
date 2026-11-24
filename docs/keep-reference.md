# Independent KEEP CSV reference for the ETL comparison

`keep_reference` independently reconstructs four deterministic boundaries from common raw OMOP and Athena CSVs: raw mapped diagnoses, ontology graphs, ancestor target sets, and integer patient co-occurrence counts. It uses Python 3.12's standard library. The domain algorithms do not import PyHealth or the historical PyHealth-backed exporter. It is labeled `keep_reconstruction`; the released KEEP repository does not supply these raw producers. The existing `keep_preprocessing` path and its released history-projection comparison remain available with their separate lineage.

## Run from the supplied KEEP checkout

No training environment is needed. `-S` disables third-party site packages and works for the reference build and bundle validator. Choose fresh input and output directories:

```sh
uv venv --python 3.12 .venv-reference
.venv-reference/bin/python -S -m keep_reference fixture --output .keep-work/common-csv
.venv-reference/bin/python -S -m keep_reference build \
  --config .keep-work/common-csv/reference.toml --output .keep-work/reference-v1
.venv-reference/bin/python -S -m keep_reference validate .keep-work/reference-v1
```

`fixture` writes generated common CSVs and a TOML config with explicit table paths. Its input generator extends the historical public fixture; sharing this input generation does not share either implementation's computation. [`examples/keep_reference.toml`](../examples/keep_reference.toml) is another explicit config for these inputs and shows an optional `expected_keep_head` pin. Set it to the selected candidate commit when coordinating an ETL comparison; the example leaves it unset so it also runs after a new commit. Every build still records the actual revision and source hashes. All paths are relative to the config file, or absolute when explicitly supplied. No config, data directory or private output is discovered automatically. TOML and JSON configs are accepted. Invalid settings and existing output directories fail.

The CLI must run with its working directory set to the supplied KEEP checkout. It verifies its imported package origin against that directory, records the actual HEAD and working-source fingerprints, and rejects an explicitly requested revision mismatch. Source fingerprinting covers all reference modules, the actually imported module files, the relevant tracked diff and untracked reference source names. Canonical manifests contain no absolute machine paths or execution clocks. The reference does not use a cache: each build reads and hashes its explicit CSV inputs again and refuses input/source changes during a run.

For an external ETL harness, invoke the reference as a subprocess, with an absolute config path and fresh output directory in the ETL project's approved storage:

```python
subprocess.run(
    [python, "-S", "-m", "keep_reference", "build",
     "--config", input_config, "--output", reference_output],
    cwd=keep_checkout, check=True, capture_output=True, text=True,
)
```

This invocation writes only the supplied output location. It does not modify the KEEP checkout, initialize Git, or touch a PyHealth/ETL checkout. CLI stdout reports execution status and schema, not patient IDs or counts. Errors omit input paths and source values. Detailed patient outputs stay in the explicitly selected bundle directory. The ETL harness should validate returned source hashes and input digests against its declared checkout and the files consumed by its PyHealth adapter.

## Input contract and matched settings

The `inputs` map names all eight explicit files: `person`, `visit_occurrence`, `death`, `condition_occurrence`, `CONCEPT`, `CONCEPT_RELATIONSHIP`, `CONCEPT_ANCESTOR`, `VOCABULARY`. OMOP files are ordinary comma-separated CSV. Athena files are tab-separated CSV with unquoted fields. The generated files and module readers show the required column names.

Patient membership comes from `person.csv`, including patients with no diagnoses. An optional config `patient_ids` list selects a common cohort and preserves selected empty patients. It does not mean a positive/negative prediction cohort. Source patient, diagnosis and visit IDs remain opaque text; diagnosis copies are keyed by patient and original occurrence ID. Exact copies collapse before mapping; conflicting copies fail. Distinct diagnoses in one visit remain distinct occurrences. Invalid CSV structure, unknown patients and unsupported settings fail.

Each history exposes original `source_value`, nullable `source_concept_id`, `mapped_concepts`, `unmapped_reason`, visit and event time. A nonzero canonical concept ID takes precedence. Otherwise `ICD9CM:`/`ICD10CM:` fallback keeps every positive target ID from valid `Maps to` relationships, with dot removal, uppercase matching and retained ICD leading zeros. Retired ICD source concepts remain usable when their relationships are valid; retired relationships remain excluded. Raw targets are not filtered by target domain, validity, standard status or vocabulary, and a positive target missing from `CONCEPT` remains visible. This matches the pinned raw PyHealth mapper. Roll-up subsequently classifies these IDs (`outside_domain`, `unknown_code`, `unreachable`, etc.) and decides which yield retained targets. `concept_filter` governs graph eligibility, not raw mapper output. Original source text remains visible. History failure states distinguish missing source, unsupported source syntax and empty ICD mapping; a raw mapping with later ineligible targets is not mislabeled an empty ICD mapping.

Event time uses nonempty `condition_start_datetime` first and `condition_start_date` only when datetime is empty. An invalid nonempty datetime fails instead of falling back. Missing times remain null; no visit-time substitution occurs. Supported times are timezone-naive ISO values represented at microsecond precision; a date becomes midnight. More than six fractional digits are rejected before parsing, including trailing zero digits and comma fractions. For example, `.1234561` and `.1234569` both fail rather than collapsing into `.123456`. Up to six fractional digits are preserved exactly. The fixture includes a subsecond datetime, a conflicting date to prove datetime precedence, a date-only fallback, and a missing time. The current ETL adapter still accepts only whole-second datetime input; microsecond support there remains ETL-owned. It must not truncate times to manufacture equality.

The baseline declares active standard SNOMED Conditions, Disease root `4274025`, minimum root separation from closure, cutoff five, root excluded, no orphan rescue, and isolates retained. Direct `Is a` edges are parent-to-child. The uncut rooted node universe includes deep sources and isolated closure-rooted nodes. Only direct edges enter direct graph files. Selected separation-one closure links are exported separately and are overlaid for `path_source="closure"`. Both selected traversal graphs are checked for cycles.

Retained sources map to themselves. `all_retained` collects every eligible ancestor for a deep source; `first_retained` stops each branch at its first eligible ancestor and can still return several targets. `path_source="direct"` isolates the direct/closure difference. Other baseline choices are fixed and explicit in the manifest. These are reconstruction settings, not established author artifact semantics.

Complete-history presence requires two distinct diagnoses resolving to a target after roll-up. Each diagnosis contributes its target union once. Every patient increments each pair of present conditions once, with both triangles and prevalence diagonals retained. Zero rows and count-one entries remain. No prediction censoring, weighting, quantile, normalization or training filter enters these counts.

## `keep-comparison-v1` interchange

The new comparison schema is separate from preparation schema 1. No existing preparation artifact was reinterpreted.

| File | Boundary and columns |
|---|---|
| `histories.jsonl` | Raw diagnoses: `patient_id`, `diagnosis_id`, `visit_id`, `event_time`, `source_value`, `source_concept_id`, `mapped_concepts`, `unmapped_reason`. No rolled-up or prediction substitution. |
| `concepts.tsv` | Retained nodes: `row_id`, `concept_id`, `label`, `root_depth`. |
| `edges.tsv` | Retained direct edges: `parent_id`, `child_id`, `relation` (`is_a`). |
| `uncut_concepts.tsv`, `uncut_edges.tsv` | Uncut traversal nodes and direct edges, with the same columns. |
| `closure_links.tsv` | Selected overlay: `ancestor_id`, `descendant_id`, `distance` (one). Overlay links are never inserted into the direct-edge exports. |
| `rollup.tsv` | Unique `source_id`, `target_id` pairs. |
| `rollup_status.tsv` | Every declared source: `source_id`, nullable `root_depth`, native `reason`, including empty target sets. |
| `rolled_histories.jsonl` | Supplementary native stage output: raw history fields plus deduplicated `rolled_up_concepts`. Enables conditional counter experiments without replacing the raw history boundary. |
| `patient_presence.jsonl` | Every selected patient: `patient_id`, qualifying `concept_ids`, including empty sets. |
| `counts.tsv` | Unique nonzero integer `row`, `column`, `count` triples. |
| `manifest.json` | Explicit settings; complete patient and source universes; independent `row_concept_ids`, `column_concept_ids`, `matrix_shape`; per-file input/output digests; revision/source identity; per-boundary producer provenance; native relation/status mappings; missing author-reference coverage. |

Concept IDs are canonical decimal strings. Patient and diagnosis IDs are not inferred from positions. The producer derives its own retained universe and native numeric ordering from the CSVs. The loader supports independent row and column permutations when values are associated correctly; it never intersects away missing concepts. Both row and column universes must match the full retained graph. The complete roll-up source universe includes valid standard SNOMED concepts of all domains, observed diagnosis concept IDs (including invalid/unknown IDs), and optional config `source_ids` probes.

Each boundary names its actual `producer_entry_point`, executed source fingerprints, inputs, output digests, upstream boundaries, `provenance_kind="keep_reconstruction"`, `execution_status="produced"`, and an empty `supplied_common_inputs` list for a full independent build. `author_reference_coverage` remains `missing_reference` for the four raw stages. Status vocabulary is preserved with a declared semantic mapping. The existing released positive/control projection comparison is a different, downstream stage.

New bundles also declare additive `mapping_contract` and `timestamp_contract` fields without changing the existing `settings` map or config/CLI API. The mapping contract version is `keep-reference-raw-mapping-v2`: retired ICD sources are included, only valid relationships are used, raw positive target IDs are preserved and target classification happens during roll-up. The timestamp contract declares pre-parse rejection of excess precision. These fields identify the fixes from the ETL review. Older v1 bundles lacking these declarations are loaded with their original metadata; their mapping/precision policy must not be inferred from the current defaults. ETL should inspect the declarations alongside source fingerprints. Its existing settings bridge remains usable.

`load_bundle()` checks digests and structure. `validate_bundle()` checks internal graph/axis/history/target/count consistency; it does not perform a second algorithmic implementation or establish parity with PyHealth. Treat its `internally_consistent` status separately from an ETL comparison result of `equal`, `different`, `missing_reference`, `incomparable` or `error`.

## Public stage machinery for inspection or conditional comparisons

```python
from keep_reference import compute, load_config, load_bundle
from keep_reference.ontology import load_ontology
from keep_reference.histories import read_histories
from keep_reference.rollup import source_universe, rollup_sources, rollup_histories
from keep_reference.counts import count_cooccurrence

config = load_config(explicit_config_path)
result = compute(config)  # Native structures, no bundle writes.
result.histories
result.ontology.depths, result.ontology.direct_edges, result.ontology.retained_ids
result.ontology.closure_links
result.rollup_results
result.patient_presence, result.counts

reference = load_bundle(explicit_bundle_path)
reference.histories
reference.tables["uncut_edges.tsv"]
reference.counts_by_concept()  # Decodes each native axis separately.
```

For a conditional counting experiment, `count_cooccurrence(shared_rolled_histories, patient_ids, row_concept_ids, column_concept_ids=None)` returns patient presence and sparse triples. The records require source patient/diagnosis identities and rolled-up targets; all extra raw fields are preserved for conflicting-copy checks. Explicitly label the supplied stage inputs and their producer in the ETL report. Passing this comparison checks that counting stage, not independent mapping or roll-up. The full build never accepts a PyHealth-produced graph, vocabulary axis, targets or count matrix.

## Source trace, evidence and ETL responsibilities

Hand-derived expectations are in [`tests/fixtures/reference_expected.toml`](../tests/fixtures/reference_expected.toml). Run reference checks without any training work:

```sh
.venv-preprocessing/bin/python -m pytest -q tests/test_reference.py
.venv-preprocessing/bin/ruff check keep_reference tests/test_reference.py
```

The current reference and affected nontraining preparation-contract checks are recorded separately from historical training evidence. The ETL project still owns its PyHealth adapter, exact four-boundary alignment/assertions, missing-coverage results, notebook and headless report. No sibling project, private data, contribution status or Git setup was changed by this KEEP implementation. This handoff provides the KEEP side; an executed two-side notebook is still required to complete the corrected C-0014 criteria.
