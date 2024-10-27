import json
import logging
import os

from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.graph.graph import Graph, UnconnectedGraphError
from tuw_nlp.sem.hrg.common.io import get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.script.loop_script import LoopOnSenDirs
from tuw_nlp.sem.hrg.common.triplet import Triplet
from tuw_nlp.sem.hrg.train.generation.per_word import get_rules_per_word


def get_argument_graphs(triplet_graph, arguments, log):
    a_graphs = {}
    for arg, nodes in arguments.items():
        try:
            a_graph = triplet_graph.G.subgraph(nodes)
            a_graphs[arg] = a_graph
        except UnconnectedGraphError:
            log.write(
                f"unconnected argument ({nodes})\n"
            )
            a_graphs[arg] = None
    return a_graphs


class Train(LoopOnSenDirs):
    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json, log=True)
        self.method = self.config["method"]
        self.out_dir += f"_{self.method}"
        vocab_file = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/preproc/vocab/" \
                     f"{self.config['vocab_file']}"
        self.vocab = Vocabulary.from_file(vocab_file)
        self.unconnected_args = []
        self.no_rule = []

    def _do_for_sen(self, sen_idx, sen_dir):
        graph_files = sorted([f"{sen_dir}/{fn}" for fn in os.listdir(sen_dir) if fn.endswith("_triplet.graph")])
        for graph_file in graph_files:
            exact_sen_idx = int(graph_file.split("/")[-1].split("_triplet.graph")[0].split("sen")[-1])
            hrg_dir = f"{self.out_dir}/{str(exact_sen_idx)}"
            if not os.path.exists(hrg_dir):
                os.makedirs(hrg_dir)
            log = open(f"{hrg_dir}/sen{exact_sen_idx}.log", "w")
            print(f"Processing sen {exact_sen_idx}")

            with open(graph_file) as f:
                lines = f.readlines()
                assert len(lines) == 1
                graph_str = lines[0].strip()
            triplet_graph = Graph.from_bolinas(graph_str)
            triplet = Triplet.from_file(f"{sen_dir}/sen{exact_sen_idx}_triplet.txt")

            arg_graphs = get_argument_graphs(triplet_graph, triplet.arguments, log)

            if None in arg_graphs.values():
                log.write(f"sentence {exact_sen_idx} had unconnected arguments, skipping\n")
                self.unconnected_args.append(exact_sen_idx)
                continue

            initial_rule = ""
            rules = []
            if self.method == "per_word":
                initial_rule, rules = get_rules_per_word(triplet_graph, triplet, log)
            # Todo: per_arg
            if not initial_rule:
                assert len(rules) == 0
                self.no_rule.append(exact_sen_idx)
            else:
                with open(f"{hrg_dir}/sen{exact_sen_idx}.hrg", "w") as f:
                    f.write(f"{initial_rule}")
                    for rule in sorted(rules):
                        f.write(f"{rule}")

    def _after_loop(self):
        self._log(
            f"\nNumber of unconnected arguments: {len(self.unconnected_args)}\n"
            f"{json.dumps(self.unconnected_args)}"
        )
        self._log(
            f"\nNumber of no rules: {len(self.no_rule)}\n"
            f"{json.dumps(self.no_rule)}",
        )
        super()._after_loop()


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)
    args = get_data_dir_and_config_args("Script to create hrg rules on preprocessed train data.")
    script = Train(
        args.data_dir,
        args.config,
    )
    script.run()
