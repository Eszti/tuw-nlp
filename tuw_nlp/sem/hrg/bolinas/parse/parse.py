import json
import os.path
import fileinput
import pickle
import time

from argparse import ArgumentParser
from datetime import datetime


from tuw_nlp.sem.hrg.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.bolinas.parser_basic.vo_rule import VoRule
from tuw_nlp.sem.hrg.common.io import get_range, log_to_console_and_log_lines


def load_grammar(grammar_file, backward, nodelabels, logprob, log_lines):
    with open(grammar_file) as f:
        grammar = Grammar.load_from_file(f, VoRule, backward, nodelabels=nodelabels, logprob=logprob)
    rhs2_type = f"-to-{grammar.rhs2_type}" if grammar.rhs2_type else ''
    log_str = f"\nLoaded {grammar.rhs1_type}{rhs2_type} grammar with {len(grammar)} rules."
    log_to_console_and_log_lines(log_str, log_lines)
    return grammar


def parse_sen(graph_parser, graph_file, chart_file, sen_log_file, max_steps):
    sen_log_lines = []
    parse_generator = graph_parser.parse_graphs(
        (Hgraph.from_string(x) for x in fileinput.input(graph_file)),
        sen_log_lines,
        partial=True,
        max_steps=max_steps
    )
    for i, chart in enumerate(parse_generator):
        assert i == 0
        if "START" not in chart:
            log_to_console_and_log_lines("No derivation found", sen_log_lines)
            continue
        else:
            log_to_console_and_log_lines(f"Chart len: {len(chart)}", sen_log_lines)
            with open(chart_file, "wb") as f:
                pickle.dump(chart, f, -1)
    with open(sen_log_file, "w") as f:
        f.writelines(sen_log_lines)


def main(data_dir, config_json):
    start_time = time.time()

    logprob = True
    nodelabels = True
    backward = False

    if not config_json:
        config_json = f"{os.path.dirname(os.path.realpath(__file__))}/parse_config.json"
    config = json.load(open(config_json))

    log_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "log",
        "parse_" + config["out_dir"] + ".log"
    )
    log_lines = [f"Execution start: {datetime.now()}\n"]
    first = config.get("first", None)
    last = config.get("last", None)
    if first:
        log_to_console_and_log_lines(f"First: {first}", log_lines)
    if last:
        log_to_console_and_log_lines(f"Last: {last}", log_lines)

    max_steps = config.get("max_steps", 10000)
    log_to_console_and_log_lines(f"Max steps: {max_steps}", log_lines)

    grammar_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}//train/grammar/"
    grammar_file = f"{grammar_dir}/{config['grammar_file']}"
    grammar = load_grammar(grammar_file, backward, nodelabels, logprob, log_lines)

    graph_parser = Parser(grammar)

    in_dir = f"{data_dir}/{config['preproc_dir']}"
    out_dir = f"{data_dir}/{config['out_dir']}"
    first_sen_to_proc = None
    last_sen_to_proc = None
    for sen_idx in get_range(in_dir, first, last):
        if first_sen_to_proc is None:
            first_sen_to_proc = sen_idx

        print(f"\nProcessing sen {sen_idx}\n")
        preproc_dir = f"{in_dir}/{str(sen_idx)}/preproc"
        graph_file = f"{preproc_dir}/pos_edge.graph"

        bolinas_dir = f"{out_dir}/{str(sen_idx)}/bolinas"
        if not os.path.exists(bolinas_dir):
            os.makedirs(bolinas_dir)
        chart_file = f"{bolinas_dir}/sen{str(sen_idx)}_chart.pickle"
        sen_log_file = f"{bolinas_dir}/sen{str(sen_idx)}_parse.log"

        parse_sen(graph_parser, graph_file, chart_file, sen_log_file, max_steps)
        last_sen_to_proc = sen_idx

    log_to_console_and_log_lines(f"\nFirst sentence to process: {first_sen_to_proc}", log_lines)
    log_to_console_and_log_lines(f"Last sentence to process: {last_sen_to_proc}", log_lines)

    log_to_console_and_log_lines(f"\nExecution finish: {datetime.now()}", log_lines)
    elapsed_time = time.time() - start_time
    time_str = f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n"
    log_to_console_and_log_lines(time_str, log_lines)
    with open(log_file, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description ="Parse graph inputs and save chart.")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    main(args.data_dir, args.config)
