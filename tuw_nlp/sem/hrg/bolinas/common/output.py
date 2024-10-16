import re

from tuw_nlp.sem.hrg.bolinas.common.hgraph.hgraph import Hgraph


def print_shifted(derivation):
    final_item = derivation[1]["START"][0]
    node_to_concepts = dict(zip(final_item.nodeset, [""]*len(final_item.nodeset)))
    triples = []
    for v, l, u in final_item.shifted:
        triples.append((v[0], l, u[0][0]))
    graph = Hgraph.from_triples(triples, node_to_concepts)
    return re.sub(r"(\n|\s+)", " ", graph.to_bolinas_str(nodeids=True))


def walk_derivation(derivation, combiner, leaf):
    if type(derivation) is not tuple:
        if derivation == "START":
            return None
        return leaf(derivation)
    else:
        item, children = derivation[0], derivation[1]
        childobjs = dict([(rel, walk_derivation(c, combiner, leaf)) for (rel, c) in children.items()])

        if item == "START":
            return childobjs["START"]

        return combiner(item, childobjs)


def format_derivation(derivation):
    def combiner(item, childobjs):
        children = []
        for nt, child in childobjs.items():
            edgestring = "$".join(nt)
            children.append("%s(%s)" % (edgestring, child))
        childstr = " ".join(children)
        return "%s(%s)" % (item.rule.rule_id, childstr)

    def leaf(item):
        return str(item.rule.rule_id)

    return walk_derivation(derivation, combiner, leaf)
