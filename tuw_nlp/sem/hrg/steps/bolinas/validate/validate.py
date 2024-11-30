from tuw_nlp.sem.hrg.common.io import log_to_console_and_log_lines
from tuw_nlp.sem.hrg.steps.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.common.oie import extract_for_kth_derivation
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_rule import VoRule


def check_if_graph_accepted_by_hrg(grammar_lines, bolinas_graph):
    accepted = False
    log_lines = ["\nVALIDATION:\n\nParsing:\n"]
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
        log_lines.append("\nSearch:\n")
        log_to_console_and_log_lines(f"Chart keys len: {len(chart)}", log_lines)
        log_to_console_and_log_lines(f"Chart items len: {chart.items_length()}", log_lines)
        if "START" not in chart:
            log_lines.append("No derivation found")
        else:
            first_derivation, steps = chart.first_derivation("START")
            log_to_console_and_log_lines(f"Steps for search: {steps}", log_lines)
            log_lines.append("\nDerivation:\n")
            score = first_derivation[0]
            derivation = first_derivation[1]
            shifted_derivation, used_rules, matched_nodes = extract_for_kth_derivation(
                derivation,
                score,
                i+1,
            )
            log_lines.append(f"{shifted_derivation}\n")
            log_lines.append(f"{used_rules}\n")
            log_lines.append(f"{matched_nodes}\n")
            accepted = True
    return log_lines, accepted
