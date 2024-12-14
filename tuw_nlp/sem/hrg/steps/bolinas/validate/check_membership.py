from tuw_nlp.sem.hrg.steps.bolinas.common.exceptions import NotAllNodesCoveredException
from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.common.oie import extract_for_kth_derivation


def check_membership(parser, bolinas_graph):
    used_rules = None
    log_lines = ["\nVALIDATION:\n"]
    input_graph = Hgraph.from_string(bolinas_graph)
    orig_nodes = sorted(list(input_graph.get_nodes().keys()), key=lambda node: int(node[1:]))
    parse_generator = parser.parse_graphs([input_graph], partial=False)

    for i, (chart, parse_logs) in enumerate(parse_generator):
        assert i == 0
        log_lines.append(f"\n{parse_logs}")
        if "START" not in chart:
            log_lines.append("No derivation found\n")
        else:
            derivations, search_log = chart.derivations("START", only_first=True)
            log_lines.append(f"\n{search_log}")

            derivation_log, used_rules, derived_nodes = extract_for_kth_derivation(
                derivation=derivations[0][1],
                n_score=derivations[0][0],
                ki=i+1,
            )
            log_lines.append(f"\n{derivation_log}")

            not_covered_nodes = sorted(set(orig_nodes) - set(derived_nodes), key=lambda node: int(node[1:]))
            if len(not_covered_nodes) != 0:
                raise NotAllNodesCoveredException(orig_nodes, derived_nodes, not_covered_nodes)

    return log_lines, used_rules
