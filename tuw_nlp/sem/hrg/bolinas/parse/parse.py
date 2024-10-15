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
from tuw_nlp.sem.hrg.common.io import get_range


def load_grammar(grammar_file, backward, nodelabels, logprob):
    with open(grammar_file) as f:
        grammar = Grammar.load_from_file(f, VoRule, backward, nodelabels=nodelabels, logprob=logprob)
    print("Loaded %s%s grammar with %i rules." % 
          (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar)))
    return grammar


def parse_sen(graph_parser, graph_file, chart_file):
    parse_generator = graph_parser.parse_graphs(
        (Hgraph.from_string(x) for x in fileinput.input(graph_file)), partial=True, max_steps=10000)
    for i, chart in enumerate(parse_generator):
        assert i == 0
        if "START" not in chart:
            print("No derivation found")
            continue
        else:
            print("Chart len: %d" % len(chart))
            with open(chart_file, "wb") as f:
                pickle.dump(chart, f, -1)


def main(data_dir):
    start_time = time.time()

    logprob = True
    nodelabels = True
    backward = False

    config_json = f"{os.path.dirname(os.path.realpath(__file__))}/parse_config.json"
    config = json.load(open(config_json))

    log_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "log",
        "parse_" + config["out_dir"] + ".log"
    )
    log_lines = ["Execution start: %s\n" % str(datetime.now())]
    first = config.get("first", None)
    last = config.get("last", None)

    grammar_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}//train/grammar/"
    grammar_file = f"{grammar_dir}/{config['grammar_file']}"
    grammar = load_grammar(grammar_file, backward, nodelabels, logprob)

    graph_parser = Parser(grammar)

    in_dir = os.path.join(data_dir, config["preproc_dir"])
    out_dir = os.path.join(data_dir, config["out_dir"])
    for sen_idx in get_range(in_dir, first, last):
        print(f"\nProcessing sen {sen_idx}\n")
        sen_dir_in = f"{in_dir}/{str(sen_idx)}"
        sen_dir_out = f"{out_dir}/{str(sen_idx)}"
        preproc_dir = f"{sen_dir_in}/preproc"
        graph_file = f"{preproc_dir}/pos_edge.graph"

        bolinas_dir = f"{sen_dir_out}/bolinas"
        if not os.path.exists(bolinas_dir):
            os.makedirs(bolinas_dir)
        chart_file = f"{bolinas_dir}/sen{str(sen_idx)}_chart.pickle"

        parse_sen(graph_parser, graph_file, chart_file)

    log_lines.append("Execution finish: %s\n" % str(datetime.now()))
    elapsed_time = time.time() - start_time
    time_str = "Elapsed time: %d min %d sec" % (elapsed_time / 60, elapsed_time % 60)
    print(time_str)
    log_lines.append(time_str)
    with open(log_file, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description ="Parse graph inputs and save chart.")
    parser.add_argument("-d", "--data-dir", type=str)
    args = parser.parse_args()

    main(args.data_dir)
