import ast
import runpy
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from keep_preprocessing.build import omop_history, read_cohort
from keep_preprocessing.bundle import count_histories
from keep_preprocessing.consumers import align_vectors, full_matrix_quantile, training_triples
from keep_preprocessing.projection import project, timestamp


def row(identity, time, targets, patient="p"):
    return {
        "patient_id": patient,
        "diagnosis_id": identity,
        "event_time": timestamp(time),
        "rolled_up_concepts": targets,
    }


def test_post_rollup_threshold_is_cross_diagnosis_and_cross_visit():
    rows = [row("d1", "2020-01-01", ["100", "300"]), row("d2", "2020-01-02", ["100", "300"])]
    matrix, present = count_histories(rows, ["100", "200", "300"])
    assert matrix == {(0, 0): 1, (0, 2): 1, (2, 0): 1, (2, 2): 1} and present == 1
    assert count_histories(rows[:1], ["100", "200", "300"])[0] == {}
    # Distinct source codes can establish one rolled target. A rule that
    # thresholds each original diagnosis code first would discard both.
    original_codes = Counter(["source-X", "source-Y"])
    assert not {code for code, n in original_codes.items() if n >= 2}
    assert matrix  # The deliberately incorrect pre-roll-up rule differs.


def test_duplicate_copy_cannot_establish_presence():
    one = row("d1", "2020-01-01", ["100", "300"])
    with pytest.raises(ValueError, match="Duplicate diagnosis"):
        count_histories([one, one], ["100", "300"])


def test_strict_positive_cutoff_and_uncensored_controls():
    records = [
        row("before", "2020-01-01T23:59:59", ["100", "200"]),
        row("at", "2020-01-02", ["100"]),
        row("after", "2020-01-02T00:00:01", ["100"]),
        row("earlier", "2020-01-01", ["200"]),
    ]
    cutoff = timestamp("2020-01-02")
    assert project(records, {"p": (1, cutoff)}) == [
        {"patient_id": "p", "label": 1, "concept_ids": ["200"]}
    ]
    assert project(records, {"p": (0, None)}) == [
        {"patient_id": "p", "label": 0, "concept_ids": ["200", "100"]}
    ]


def test_missing_positive_time_is_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        project([row("d", None, ["100"])], {"p": (1, timestamp("2020-01-02"))})
    with pytest.raises(ValueError, match="timezone"):
        timestamp("2020-01-01T00:00:00+00:00")


class Mapper:
    def map(self, code, version):
        assert isinstance(code, str)
        return [900] if code == "0010" and version == 9 else []


def event(identity="d", concept=0, source="ICD9CM:0010", visit="v", time=None):
    return SimpleNamespace(
        condition_occurrence_id=identity,
        condition_concept_id=concept,
        condition_source_value=source,
        visit_occurrence_id=visit,
        timestamp=time or datetime(2020, 1, 1),
    )


def patient(events):
    return SimpleNamespace(patient_id="p", get_events=lambda event_type: events)


def test_omop_identity_precedence_and_leading_zeros():
    one = event()
    history = omop_history(
        patient([one, one, event(identity="d2", concept=300, source="bad", visit=None)]), Mapper()
    )
    assert len(history["conditions"]) == 2
    assert history["conditions"][0]["source_value"] == "ICD9CM:0010"
    assert history["conditions"][0]["concept_ids"] == ["900"]
    assert history["conditions"][1]["concept_ids"] == ["300"]
    assert history["conditions"][1]["visit_id"] is None
    with pytest.raises(ValueError, match="conflicting"):
        omop_history(patient([one, event(concept=300)]), Mapper())
    with pytest.raises(ValueError, match="required"):
        omop_history(patient([event(identity=None)]), Mapper())


@pytest.mark.parametrize(
    "source,reason",
    [
        (None, "missing_source"),
        ("BAD:0010", "unsupported_source_format"),
        ("ICD10CM:NOMAP", "empty_icd_mapping"),
    ],
)
def test_mapping_loss_reasons(source, reason):
    record = omop_history(patient([event(source=source)]), Mapper())["conditions"][0]
    assert record["unmapped_reason"] == reason and not record["concept_ids"]


@pytest.mark.parametrize(
    "size,triples", [(3, [(0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 4)]), (1, [(0, 0, 2)]), (2, [])]
)
def test_sparse_quantile_matches_numpy_over_implicit_zeros(size, triples):
    dense = np.zeros((size, size))
    for i, j, n in triples:
        dense[i, j] = n
    for q in [0, 0.25, 0.5, 0.75, 0.99, 1]:
        assert full_matrix_quantile(triples, size, q) == pytest.approx(
            np.quantile(dense.ravel(), q)
        )


def test_training_filters_do_not_mutate_raw_counts():
    raw = [(0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
    assert training_triples(raw, drop_count_one=True, diagonal="drop") == [(0, 1, 2), (1, 0, 2)]
    assert len(raw) == 4 and raw[0][2] == 1


def test_node2vec_sentinel_vectors_ignore_gensim_key_order():
    class Vectors(dict):
        vector_size = 2

        def get_vector(self, key):
            return self[key]

    keyed = Vectors({"2": np.array([20, 21]), "0": np.array([0, 1]), "1": np.array([10, 11])})
    assert align_vectors(keyed, 3, {1}).tolist() == [[0, 1], [0, 0], [20, 21]]
    del keyed["2"]
    with pytest.raises(ValueError, match="non-isolated"):
        align_vectors(keyed, 3, {1})


def test_projection_cohort_requires_explicit_positive_date(tmp_path):
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("patient_id,label,index_date\np,1,\n")
    with pytest.raises(ValueError, match="index date"):
        read_cohort(cohort)


def test_empty_matrix_oracle_and_fitting_cohort():
    assert count_histories([], ["100", "200"])[0] == {}
    assert count_histories([], [])[0] == {}
    records = [row("1", None, ["100"]), row("2", None, ["100"])]
    assert count_histories(records, ["100"], set())[0] == {}


def test_sparse_quantile_invalid_inputs():
    with pytest.raises(ValueError):
        full_matrix_quantile([], 0, 0.5)
    with pytest.raises(ValueError):
        full_matrix_quantile([], 2, 2)


def test_released_glove_checkpoint_and_objective_compatibility():
    import torch

    from keep_preprocessing import KEEP_BASE

    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", f"{KEEP_BASE}:trained_embeddings/our_embeddings/train_glove.py"],
        cwd=root,
        text=True,
    )
    # Compile only the original model class: the author module's top-level
    # cluster chdir is outside this comparison and must not execute.
    tree = ast.parse(source)
    original_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Glove"
    )
    namespace = {"torch": torch, "nn": torch.nn, "np": np, "device": torch.device("cpu")}
    exec(
        compile(
            ast.Module(body=[original_class], type_ignores=[]), "pinned_glove_class.py", "exec"
        ),
        namespace,
    )
    candidate_class = runpy.run_path(
        str(root / "trained_embeddings/our_embeddings/train_glove.py")
    )["Glove"]
    initial = np.arange(6, dtype=np.float32).reshape(3, 2) / 10
    original = namespace["Glove"](3, 2, initial, 1e-5)
    candidate = candidate_class(3, 2, initial, 1e-5).cpu()
    candidate.load_state_dict(original.state_dict(), strict=True)
    inputs = (
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([0.2, 0.3]),
    )
    old_loss, old_reg = original(*inputs)
    new_loss, new_reg = candidate(*inputs)
    assert torch.equal(new_loss, old_loss) and torch.equal(new_reg, old_reg)
