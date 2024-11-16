import json
import logging
import os

import networkx as nx

from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.common.triplet import Triplet
from tuw_nlp.sem.hrg.steps.train.rule_generation.per_word import get_rules_per_word


def get_argument_graphs(triplet_graph, arguments, log):
    a_graphs = {}
    for arg, nodes in arguments.items():
        nodes = [f"n{n}" for n in nodes]
        for n in nodes:
            assert n in triplet_graph.G.nodes()
        a_graph = triplet_graph.G.subgraph(nodes)
        if nx.is_weakly_connected(a_graph):
            a_graphs[arg] = a_graph
        else:
            log.write(
                f"unconnected argument ({nodes})\n"
            )
            a_graphs[arg] = None
    return a_graphs


class Train(LoopOnSenDirs):
    def __init__(self, config=None):
        super().__init__(description="Script to create hrg rules on preprocessed train data.", config=config)
        self.method = self.config["method"]
        self.out_dir += f"_{self.method}"
        vocab_file = f"{self.script_output_root}/vocab/{self.config['vocab_file']}"
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

            arg_graphs = get_argument_graphs(triplet_graph, triplet.arguments(), log)

            if None in arg_graphs.values():
                log.write(f"sentence {exact_sen_idx} had unconnected arguments, skipping\n")
                self.unconnected_args.append(exact_sen_idx)
                continue

            initial_rule = ""
            rules = []
            if self.method == "per_word":
                initial_rule, rules = get_rules_per_word(triplet_graph, triplet, log)
            # Todo: per_arg
            if initial_rule is None:
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
    Train().run()
