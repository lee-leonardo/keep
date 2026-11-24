"""Independent multi-parent ancestor traversal and diagnosis target projection."""

from collections import defaultdict

from .config import checked_settings


def source_universe(ontology, histories, additional=()):
    declared = {
        code
        for code, row in ontology.concepts.items()
        if row["vocabulary_id"] == "SNOMED"
        and row["standard_concept"] == "S"
        and not row["invalid_reason"].strip()
    }
    observed = {code for row in histories for code in row["mapped_concepts"]}
    return tuple(sorted(declared | observed | set(additional), key=int)), tuple(
        sorted(observed, key=int)
    )


def rollup_sources(ontology, source_ids, settings):
    settings = checked_settings(settings)
    retained = set(ontology.retained_ids)
    parents = defaultdict(set)
    for parent, child in ontology.direct_edges:
        parents[child].add(parent)
    if settings["path_source"] == "closure":
        for ancestor, descendant, _ in ontology.closure_links:
            parents[descendant].add(ancestor)
    results = {}
    for source in source_ids:
        metadata = ontology.concepts.get(source)
        depth = ontology.depths.get(source)
        if (
            not metadata
            or metadata["vocabulary_id"] != "SNOMED"
            or metadata["standard_concept"] != "S"
            or metadata["invalid_reason"].strip()
        ):
            reason, targets = "unknown_code", set()
        elif metadata["domain_id"] != "Condition":
            reason, targets = "outside_domain", set()
        elif depth is None:
            reason, targets = "unreachable", set()
        elif source in retained:
            reason, targets = "retained", {source}
        else:
            pending, visited, targets = list(parents[source]), set(), set()
            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node in retained:
                    targets.add(node)
                    if settings["target_policy"] == "first_retained":
                        continue
                pending.extend(parents[node])
            reason = "rolled_up" if targets else "no_target"
        results[source] = {
            "source_id": source,
            "root_depth": depth,
            "reason": reason,
            "target_ids": sorted(targets, key=int),
        }
    return results


def rollup_histories(histories, mappings):
    result = []
    for record in histories:
        targets = set()
        for source in record["mapped_concepts"]:
            if source not in mappings:
                raise ValueError("An observed diagnosis source has no roll-up result")
            targets.update(mappings[source]["target_ids"])
        result.append(record | {"rolled_up_concepts": sorted(targets, key=int)})
    return result
