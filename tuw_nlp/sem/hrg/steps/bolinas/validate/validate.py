from tuw_nlp.sem.hrg.steps.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.common.oie import extract_for_kth_derivation
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_rule import VoRule


def check_if_graph_accepted_by_hrg(grammar_lines, bolinas_graph):
    accepted = False
    log_lines = ["\nValidation:\n\nParsing:\n"]
    logprob = True
    nodelabels = True
    backward = False
    grammar = Grammar.load_from_file(grammar_lines, VoRule, backward, nodelabels=nodelabels, logprob=logprob)
    parser = Parser(grammar)
    parse_generator = parser.parse_graphs(
        [Hgraph.from_string(bolinas_graph)],
        log_lines,
    )

    for i, chart in enumerate(parse_generator):
        assert i == 0
        log_lines.append("\nDerivation result:\n")
        if "START" not in chart:
            log_lines.append("No derivation found")
        else:
            best = chart.derivations("START")[0]
            score = best[0]
            derivation = best[1]
            shifted_derivation, used_rules, matched_nodes = extract_for_kth_derivation(
                derivation,
                score,
                i+1,
                grammar_lines
            )
            log_lines.append(f"{shifted_derivation}\n")
            log_lines.append(f"{used_rules}\n")
            log_lines.append(f"{matched_nodes}\n")
            accepted = True
    return log_lines, accepted
