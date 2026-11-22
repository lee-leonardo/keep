# PyHealth-backed KEEP preparation

This document describes the historical preparation path, which uses PyHealth internally. For the independent CSV reference and four-boundary ETL interchange, use [the KEEP reference interface](keep-reference.md). Preparation fixtures and consumer compatibility do not constitute an independent KEEP-versus-PyHealth comparison.

The fork prepares graph, roll-up, diagnosis history and raw patient co-occurrence artifacts through one explicit concept index. `keep_preprocessing` uses the public APIs at [PyHealth `46261db`](https://github.com/lee-leonardo/PyHealth/tree/46261db82f68ecd5820c204f333f288a6720ce13). The author comparison uses the untouched history functions at [KEEP `89cb482`](https://github.com/G2Lab/keep/tree/89cb4823212d1a89cd57bb717731b68e7f10c789).

## Install the two CPU environments

Run from this checkout using Python 3.12 and `uv`. PyHealth requires NumPy 2; Node2Vec 0.5.0 declares NumPy below 2. Keep their dependencies in separate environments and exchange only the versioned bundle and indexed `.npz` vectors.

```sh
uv venv --python 3.12 .venv-preprocessing
uv pip sync --python .venv-preprocessing/bin/python requirements-preprocessing.lock
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-training.lock
```

The locks record the tested macOS arm64 environment. CPU preprocessing was verified with one PyHealth worker. PyHealth's Dask reader needs local loopback sockets. Training uses one Node2Vec worker and CPU PyTorch. Large-data resource use and other platforms have not been measured. The commands work directly from the checkout; an editable package install is optional.

## Generate and prepare the public fixture

Choose fresh output directories. Every row below is generated data; no Athena download or patient records are needed.

```sh
.venv-preprocessing/bin/python -m keep_preprocessing fixture --output .keep-work/example-input
.venv-preprocessing/bin/python -m keep_preprocessing build \
  --input-format omop \
  --source .keep-work/example-input/omop \
  --vocabulary .keep-work/example-input/vocabulary \
  --vocabulary-release synthetic-c0014-v1 \
  --cohort .keep-work/example-input/cohort.csv \
  --cache-dir .keep-work/example-cache \
  --output .keep-work/example-bundle
.venv-preprocessing/bin/python -m keep_preprocessing validate .keep-work/example-bundle
```

For the equivalent MIMIC-IV invocation, use `--input-format mimic4`, `--source .keep-work/example-input/mimic4`, and a fresh output directory. Both formats use the same generated vocabulary. `fixture --shuffle-seed 77` changes source row order. `--count-one` adds a patient with two Z diagnoses, producing a raw prevalence diagonal of one.

`build` reads only the explicitly named source and vocabulary locations. For OMOP, supply `person.csv`, `visit_occurrence.csv`, `death.csv`, and `condition_occurrence.csv` with the fields defined in `omop_config()` in `keep_preprocessing/build.py`. For MIMIC-IV, supply `hosp/patients.csv.gz`, `hosp/admissions.csv.gz`, `hosp/diagnoses_icd.csv.gz`, and `icu/icustays.csv.gz`, following the pinned public `mimic4_keep.yaml`. MIMIC diagnosis times are admission times, not independently observed onset times. A diagnosis without a visit remains a diagnosis; missing positive-case times are rejected when projecting prediction histories.

Vocabulary inputs are Athena-format tab-separated `CONCEPT.csv`, `CONCEPT_RELATIONSHIP.csv`, and `CONCEPT_ANCESTOR.csv`. If provided, `VOCABULARY.csv` is fingerprinted too. An explicit release label and checksums identify the supplied bundle. No private input discovery or downloading is performed. Caches and output bundles contain patient-level information when private inputs are supplied; choose their locations explicitly. `.keep-work/` is for generated local examples.

## Artifact contract

| Artifact | Contents |
|---|---|
| `concepts.tsv` | Contiguous zero-based row IDs; OMOP concept IDs sorted numerically; names and minimum root depths |
| `edges.tsv` | Direct retained parent-to-child edges; graph is directed; all indexed concepts remain nodes, including isolates |
| `rollup.tsv` | Unique source/target pairs, sorted numerically |
| `rollup_status.tsv` | Source depth and retained, rolled-up, or failure classification |
| `histories.jsonl` | Unique patient-scoped diagnosis identities, visits, canonical times, source values, mapped concepts, distinct targets and mapping outcomes |
| `counts.tsv` | Sorted nonzero `(row, column, integer count)` triples; shape includes zero rows |
| `prediction_histories.jsonl` | Optional cohort projection; included patient labels and qualifying concept sequences |
| `manifest.json` | Schema, source and artifact checksums, revisions, runtime versions, graph and counting choices, cohorts, shape and diagnostics |

The baseline uses active standard SNOMED Condition concepts, Disease root 4274025, minimum root separation, cutoff five, root exclusion, no orphan rescue, and closure-backed `all_retained` roll-up. A retained source maps to itself. Deep sources and intermediates remain available for traversal. Direct edges define the embedding graph; closure-only links can aid roll-up without becoming graph edges. All target lists are deduplicated within a source diagnosis.

Presence requires two distinct source diagnoses after roll-up, over complete patient histories. Each patient increments each present pair once. Both matrix triangles are stored; diagonals are patient prevalence. Identical diagnosis copies collapse; conflicting copies fail. MIMIC-IV identities retain admission, sequence, original ICD text and version. Distinct rows in the same visit remain distinct occurrences.

`build --fit-patients FILE` accepts a newline-separated, distinct list of known patient IDs. Only those patients fit empirical counts; histories remain available for projection. Freeze this selection before a held-out evaluation. `--cohort FILE` accepts CSV columns `patient_id,label,index_date`. Positive labels require an explicit ISO index date; their histories use events strictly before that date. Controls are uncensored and must leave the date empty. Thresholding follows censoring. Included concepts are ordered by second qualifying occurrence time, then numeric ID. Cohort derivation, random truncation and classifier evaluation are outside this preprocessing contribution.

The cache namespace fingerprints inputs, vocabulary and policy, keeping stale mapping or reader outputs separate. Output payloads exclude paths, runtime clocks and performance measurements. Source-row permutation changes source checksums while preserving canonical payloads. Validation checks integrity, shared indexing, target unions, exact patient counts and prediction histories. The manifest is published last in a staged bundle, and the destination becomes visible only after successful validation. Existing bundle directories are refused.

## Use the released consumers

```sh
.venv/bin/python trained_embeddings/our_embeddings/train_node2vec.py \
  --bundle .keep-work/example-bundle --output .keep-work/node2vec.npz \
  --dimensions 8 --walk-length 5 --num-walks 4 --epochs 2 --seed 19 --sample 0
.venv/bin/python trained_embeddings/our_embeddings/train_glove.py \
  --bundle .keep-work/example-bundle --init .keep-work/node2vec.npz \
  --output .keep-work/glove.npz --dimensions 8 --epochs 2 --seed 19
```

These are small compatibility runs. Node2Vec uses the exported directed topology, retains isolates explicitly, and exports their vectors as zero. Other vectors are retrieved by row-ID key, independently of Gensim's internal vocabulary order. `.npz` exports carry concept IDs, row IDs and graph/index fingerprints; GloVe checks them before using initialization.

GloVe consumes sorted positive sparse triples directly, using the released `Glove` objective and Adagrad optimizer. `--drop-count-one` and `--diagonal drop` alter only the training view. The raw bundle stays intact. The optional quantile uses the entire raw square matrix, including implicit zeros, before filtering; `x_max` is the maximum of the explicit floor and this quantile. The legacy default floor is 50. Omit `--init` to train GloVe without initialization or regularization. Seed, hyperparameters and package versions are recorded beside vectors.

With no arguments, each original training script retains its legacy artifact paths and constants relative to the working directory. Run the legacy command from the directory containing those artifacts. The scripts no longer change into an author-specific cluster directory at import time. Existing legacy GloVe state dictionaries remain loadable: the initial-vector buffer is nonpersistent.

## Rerun the validation evidence

```sh
.venv-preprocessing/bin/python scripts/validate_c0014.py \
  --training-python .venv/bin/python --output .keep-work/c0014-review
.venv-preprocessing/bin/ruff check keep_preprocessing tests scripts
```

The review command runs generated-input integration and contract checks, separate-process rebuilds, pinned author-history comparisons, and CPU consumers. It produces logs, JUnit results and a sanitized `summary.json` in the fresh review directory.

## What this establishes

The fixtures test the reconstruction against explicit expected values. The untouched author history functions are a differential reference only for their supported shared input. Exact author graph, target and matrix parity still requires matched author-produced artifacts. Tiny finite-loss training establishes consumer compatibility, not published model outcomes. The [manuscript algorithm](https://arxiv.org/html/2510.05049v1#A1) specifies AdamW with squared-distance regularization; the released GloVe script uses Adagrad with cosine distance by default. This fork preserves the released objective and records that distinction.
