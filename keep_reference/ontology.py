"""Independent Athena CSV parsing, ICD mappings and direct/closure graph stages."""

from collections import defaultdict
from dataclasses import dataclass

from .config import checked_settings
from .io import concept_id, read_csv


def normalize_icd(code):
    return code.strip().upper().replace(".", "")


def require_dag(nodes, edges):
    indegree = dict.fromkeys(nodes, 0)
    children = defaultdict(set)
    for parent, child in set(edges):
        if parent not in indegree or child not in indegree or parent == child:
            raise ValueError("Invalid hierarchy edge")
        children[parent].add(child)
        indegree[child] += 1
    ready = [code for code, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(indegree):
        raise ValueError("Hierarchy contains a directed cycle")


@dataclass
class Ontology:
    concepts: dict
    depths: dict[str, int]
    retained_ids: tuple[str, ...]
    direct_edges: tuple[tuple[str, str], ...]
    closure_links: tuple[tuple[str, str, int], ...]
    icd_mappings: dict

    def map_icd(self, vocabulary, code):
        return list(self.icd_mappings.get((vocabulary, normalize_icd(code)), ()))


def load_ontology(inputs, settings):
    settings = checked_settings(settings)
    concepts = {}
    for row in read_csv(
        inputs["CONCEPT"],
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "standard_concept",
            "concept_code",
            "invalid_reason",
        ],
        athena=True,
    ):
        code = concept_id(row["concept_id"], allow_zero=True)
        # OMOP's "No matching concept" sentinel is never an ontology node.
        if code == "0":
            continue
        if code in concepts:
            raise ValueError("Vocabulary concept identities must be unique")
        concepts[code] = row
    valid_standard = {
        code
        for code, r in concepts.items()
        if not r["invalid_reason"].strip()
        and r["standard_concept"] == "S"
        and r["vocabulary_id"] == "SNOMED"
    }
    condition_ids = {code for code in valid_standard if concepts[code]["domain_id"] == "Condition"}
    root = settings["root"]
    if root not in condition_ids:
        raise ValueError("Disease root is missing from active standard SNOMED Conditions")
    depths = {root: 0}
    separation_one = set()
    for row in read_csv(
        inputs["CONCEPT_ANCESTOR"],
        [
            "ancestor_concept_id",
            "descendant_concept_id",
            "min_levels_of_separation",
            "max_levels_of_separation",
        ],
        athena=True,
    ):
        ancestor, descendant = (
            concept_id(row["ancestor_concept_id"], allow_zero=True),
            concept_id(row["descendant_concept_id"], allow_zero=True),
        )
        if ancestor == "0" or descendant == "0":
            continue
        minimum, maximum = (
            int(row["min_levels_of_separation"]),
            int(row["max_levels_of_separation"]),
        )
        if (
            minimum < 0
            or maximum < minimum
            or (ancestor != descendant and minimum == 0)
            or (ancestor == descendant and (minimum != 0 or maximum != 0))
        ):
            raise ValueError("Invalid closure separation")
        if ancestor == root and descendant in condition_ids:
            depths[descendant] = min(depths.get(descendant, minimum), minimum)
        if minimum == 1:
            separation_one.add((ancestor, descendant, 1))
    direct = set()
    mapped = defaultdict(set)
    for row in read_csv(
        inputs["CONCEPT_RELATIONSHIP"],
        ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
        athena=True,
    ):
        if row["invalid_reason"].strip():
            continue
        child, parent = (
            concept_id(row["concept_id_1"], allow_zero=True),
            concept_id(row["concept_id_2"], allow_zero=True),
        )
        if child == "0" or parent == "0":
            continue
        relation = row["relationship_id"]
        if relation == "Is a" and child in depths and parent in depths and child != parent:
            direct.add((parent, child))
        # Raw mapping preserves valid relationship targets. Graph eligibility
        # and diagnosis roll-up classification are separate, later decisions.
        # Historical ICD sources may be retired while their links remain valid.
        if relation == "Maps to":
            source = concepts.get(child)
            if source and source["vocabulary_id"] in {"ICD9CM", "ICD10CM"}:
                mapped[(source["vocabulary_id"], normalize_icd(source["concept_code"]))].add(parent)
    links = {(a, d, n) for a, d, n in separation_one if a in depths and d in depths}
    require_dag(depths, direct)
    if settings["path_source"] == "closure":
        require_dag(depths, direct | {(a, d) for a, d, _ in links})
    retained = tuple(
        sorted(
            (
                code
                for code, depth in depths.items()
                if code != root and depth <= settings["max_depth"]
            ),
            key=int,
        )
    )
    # Read the declared release table as part of the explicitly supplied input
    # chain, rather than guessing a release from a local directory name.
    list(read_csv(inputs["VOCABULARY"], ["vocabulary_id", "vocabulary_version"], athena=True))

    def key(edge):
        return tuple(map(int, edge))

    return Ontology(
        concepts,
        depths,
        retained,
        tuple(sorted(direct, key=key)),
        tuple(sorted(links, key=key)),
        {k: tuple(sorted(v, key=int)) for k, v in mapped.items()},
    )
