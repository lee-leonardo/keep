"""Bundle-backed adapters for the released Node2Vec and GloVe consumers.

No PyHealth import: these consumers run in the separate training environment.
"""

import argparse
import json
import math
from pathlib import Path

from .bundle import digest, read_table, validate, write_json


def load_inputs(bundle):
    bundle = Path(bundle)
    validate(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    ids = [row["concept_id"] for row in read_table(bundle, "concepts.tsv")]
    index = {code: i for i, code in enumerate(ids)}
    edges = [(index[r["parent_id"]], index[r["child_id"]]) for r in read_table(bundle, "edges.tsv")]
    triples = [
        (int(r["row"]), int(r["column"]), int(r["count"])) for r in read_table(bundle, "counts.tsv")
    ]
    return manifest, ids, edges, triples


def graph_from_bundle(bundle):
    import networkx as nx

    _, ids, edges, _ = load_inputs(bundle)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(ids)))
    graph.add_edges_from(edges)
    return graph, ids


def align_vectors(keyed_vectors, n, isolates):
    """Lookup by row-ID key, independent of Gensim's frequency-sorted key order.

    Isolates receive explicit zero vectors. A missing non-isolate is an error.
    """
    import numpy as np

    result = np.zeros((n, keyed_vectors.vector_size), dtype=np.float32)
    for row in range(n):
        if row not in isolates:
            key = str(row)
            if key not in keyed_vectors:
                raise ValueError("Node2Vec omitted a non-isolated graph node")
            result[row] = keyed_vectors.get_vector(key)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Node2Vec vector")
    return result


def save_embeddings(output, vectors, ids, bundle):
    import numpy as np

    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Embedding output must use .npz")
    if vectors.ndim != 2 or len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding matrix")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(
            stream,
            vectors=vectors,
            concept_ids=np.asarray(ids),
            row_ids=np.arange(len(ids)),
            concepts_sha256=np.asarray(digest(Path(bundle) / "concepts.tsv")),
            edges_sha256=np.asarray(digest(Path(bundle) / "edges.tsv")),
        )


def load_embeddings(path, ids, bundle):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if archive["concept_ids"].tolist() != ids or archive["row_ids"].tolist() != list(
            range(len(ids))
        ):
            raise ValueError("Embedding concept index mismatch")
        for key, file in [("concepts_sha256", "concepts.tsv"), ("edges_sha256", "edges.tsv")]:
            if str(archive[key]) != digest(Path(bundle) / file):
                raise ValueError("Embedding artifact provenance mismatch")
        vectors = archive["vectors"].copy()
    if vectors.ndim != 2 or len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding matrix")
    return vectors


def full_matrix_quantile(triples, size, q):
    """NumPy's linear quantile over V*V entries, including implicit zeroes."""
    if not 0 <= q <= 1 or size <= 0:
        raise ValueError("Quantile requires a nonempty square matrix and q in [0,1]")
    positive = sorted(n for _, _, n in triples)
    zeros = size * size - len(positive)
    if zeros < 0:
        raise ValueError("Too many sparse entries")
    position = q * (size * size - 1)
    lo, hi = math.floor(position), math.ceil(position)

    def value(i):
        return 0 if i < zeros else positive[i - zeros]

    return value(lo) + (value(hi) - value(lo)) * (position - lo)


def training_triples(triples, *, drop_count_one=False, diagonal="keep"):
    return [
        (i, j, n)
        for i, j, n in triples
        if not (drop_count_one and n == 1) and not (diagonal == "drop" and i == j)
    ]


def node2vec_main():
    import importlib.metadata
    import random

    import networkx as nx
    import numpy as np
    from node2vec import Node2Vec

    parser = argparse.ArgumentParser(description="Train Node2Vec from a validated KEEP bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--walk-length", type=int, default=30)
    parser.add_argument("--num-walks", type=int, default=750)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample",
        type=float,
        default=0.001,
        help="Gensim downsampling; use 0 for tiny smoke fixtures",
    )
    args = parser.parse_args()
    if min(args.dimensions, args.walk_length, args.num_walks, args.epochs) < 1:
        raise ValueError("Training sizes must be positive")
    if not math.isfinite(args.sample) or args.sample < 0:
        raise ValueError("Downsampling must be finite and nonnegative")
    if Path(args.output).exists() or Path(args.output + ".json").exists():
        raise ValueError("Select a fresh output path")
    graph, ids = graph_from_bundle(args.bundle)
    if not ids:
        raise ValueError("Cannot train an empty vocabulary")
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = Node2Vec(
        graph,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=1,
        seed=args.seed,
        quiet=True,
    ).fit(
        window=10,
        min_count=1,
        batch_words=4096,
        epochs=args.epochs,
        seed=args.seed,
        workers=1,
        compute_loss=True,
        sample=args.sample,
    )
    isolates = set(nx.isolates(graph))
    vectors = align_vectors(model.wv, len(ids), isolates)
    loss = float(model.get_latest_training_loss())
    if not math.isfinite(loss):
        raise ValueError("Nonfinite Node2Vec training loss")
    save_embeddings(args.output, vectors, ids, args.bundle)
    summary = {
        "consumer": "node2vec",
        "seed": args.seed,
        "workers": 1,
        "sample": args.sample,
        "epochs": args.epochs,
        "dimensions": args.dimensions,
        "walk_length": args.walk_length,
        "num_walks": args.num_walks,
        "directed": True,
        "direction": "parent_to_child",
        "nodes": len(ids),
        "edges": graph.number_of_edges(),
        "isolate_policy": "zero",
        "isolates": len(isolates),
        "finite_vectors": True,
        "training_loss": loss,
        "bundle_artifacts": json.loads((Path(args.bundle) / "manifest.json").read_text())[
            "artifacts"
        ],
        "training_requirements_sha256": digest(
            Path(__file__).resolve().parents[1] / "requirements-training.lock"
        ),
        "packages": {
            p: importlib.metadata.version(p) for p in ["node2vec", "gensim", "numpy", "networkx"]
        },
    }
    write_json(args.output + ".json", summary)
    print(json.dumps(summary, sort_keys=True))


def glove_main(model_class):
    import importlib.metadata
    import random

    import numpy as np
    import torch

    parser = argparse.ArgumentParser(
        description="Train the released GloVe objective from sparse bundle counts"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init", help="Aligned Node2Vec .npz; omit for uninitialized GloVe")
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--x-max", type=float, default=50)
    parser.add_argument("--x-max-quantile", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--regularization", type=float, default=0.00001)
    parser.add_argument("--reg-norm", type=float)
    parser.add_argument("--drop-count-one", action="store_true")
    parser.add_argument("--diagonal", choices=["keep", "drop"], default="keep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        min(args.dimensions, args.epochs, args.batch_size) < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError("Training sizes and learning rate must be positive")
    if (
        not all(math.isfinite(value) and value > 0 for value in [args.alpha, args.x_max])
        or not math.isfinite(args.regularization)
        or args.regularization < 0
    ):
        raise ValueError("Invalid GloVe weighting or regularization")
    if args.reg_norm is not None and (not math.isfinite(args.reg_norm) or args.reg_norm <= 0):
        raise ValueError("Regularization norm must be finite and positive")
    if Path(args.output).exists() or Path(args.output + ".json").exists():
        raise ValueError("Select a fresh output path")
    _, ids, _, raw_triples = load_inputs(args.bundle)
    x_max = max(args.x_max, full_matrix_quantile(raw_triples, len(ids), args.x_max_quantile))
    triples = training_triples(
        raw_triples, drop_count_one=args.drop_count_one, diagonal=args.diagonal
    )
    if not triples:
        raise ValueError("No positive training triples under the selected policy")
    initial = load_embeddings(args.init, ids, args.bundle) if args.init else None
    if initial is not None and initial.shape[1] != args.dimensions:
        raise ValueError("Initial embedding dimension mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = model_class(
        len(ids),
        args.dimensions,
        initial,
        args.regularization if initial is not None else None,
        args.reg_norm,
    ).cpu()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    losses = []
    # Sorted positive triples are consumed directly. No V*V allocation or scan.
    for _ in range(args.epochs):
        for start in range(0, len(triples), args.batch_size):
            batch = triples[start : start + args.batch_size]
            i = torch.tensor([r[0] for r in batch], dtype=torch.long)
            j = torch.tensor([r[1] for r in batch], dtype=torch.long)
            counts = torch.tensor([r[2] for r in batch], dtype=torch.float32)
            weights = torch.clamp((counts / x_max) ** args.alpha, max=1)
            optimizer.zero_grad()
            loss, _ = model(i, j, counts, weights)
            objective = loss.mean()
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite GloVe loss")
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach()))
    vectors = ((model.embeddings_u.weight + model.embeddings_v.weight) / 2).detach().numpy()
    save_embeddings(args.output, vectors, ids, args.bundle)
    summary = {
        "consumer": "glove",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dimensions": args.dimensions,
        "alpha": args.alpha,
        "regularization": args.regularization if initial is not None else None,
        "reg_norm": args.reg_norm,
        "initialized": initial is not None,
        "optimizer": "Adagrad",
        "regularization_type": "cosine" if args.reg_norm is None else "p_norm",
        "raw_positive_triples": len(raw_triples),
        "training_triples": len(triples),
        "drop_count_one": args.drop_count_one,
        "diagonal": args.diagonal,
        "x_max": x_max,
        "x_max_quantile": args.x_max_quantile,
        "quantile_universe": "entire_raw_matrix_including_implicit_zeros",
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "finite_vectors": True,
        "bundle_artifacts": json.loads((Path(args.bundle) / "manifest.json").read_text())[
            "artifacts"
        ],
        "training_requirements_sha256": digest(
            Path(__file__).resolve().parents[1] / "requirements-training.lock"
        ),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "numpy"]},
    }
    write_json(args.output + ".json", summary)
    print(json.dumps(summary, sort_keys=True))
