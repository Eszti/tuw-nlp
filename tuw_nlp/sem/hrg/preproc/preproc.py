import json
import os
from collections import defaultdict

import networkx as nx
import stanza

from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.graph.ud_graph import UDGraph
from tuw_nlp.sem.hrg.common.io import parse_doc, save_bolinas_str, save_as_dot, get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.script.loop_script import LoopScriptOnConll
from tuw_nlp.sem.hrg.common.triplet import Triplet


def get_ud_graph(parsed_doc):
    parsed_sen = parsed_doc.sentences[0]
    return UDGraph(parsed_sen)


def get_triplet(sen):
    arguments = defaultdict(list)
    predicate = []
    for i, tok in enumerate(sen):
        label = tok[7].split("-")[0]
        if label == "P":
            predicate.append(i + 1)
        elif label.startswith("A"):
            arguments[label].append(i + 1)
    return Triplet(predicate, arguments)


def get_triplet_subgraph(ud_graph, triplet, vocab):
    idx_to_keep = [n for nodes in triplet.arguments.values() for n in nodes] + triplet.predicate
    return ud_graph.subgraph(idx_to_keep, handle_unconnected="shortest_path").pos_edge_graph(vocab)


def add_triplet_data_to_node_name(graph, triplet, node_prefix=""):
    for n in graph.G.nodes:
        key = n
        if node_prefix:
            key = n.split(node_prefix)[1]
        label = triplet.get_label(key)
        if label:
            new_name = graph.G.nodes[n]["name"]
            if new_name:
                new_name += "\n"
            new_name += f"{label}"
            graph.G.nodes[n]["name"] = new_name


def save_conll(sen, fn):
    with open(fn, 'w') as f:
        for line in sen:
            line[0] = str(int(line[0]) + 1)
            f.write("\t".join(line))
            f.write("\n")


def add_node_labels(bolinas_graph):
    for node, data in bolinas_graph.G.nodes(data=True):
        name = data['name']
        if not name:
            data["name"] = node


class PreprocScript(LoopScriptOnConll):

    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json)
        vocab_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(config_json)))}/vocab"
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        self.vocab_file = f"{vocab_dir}/{config_json.split('/')[-1].split('.json')[0]}.txt"

    def _before_loop(self):
        self.nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,mwt,pos,lemma,depparse",
            tokenize_pretokenized=True,
        )
        self.vocab = Vocabulary(first_id=1000)

    def _do_for_sen(self, sen_idx, sen_txt, sen, last_sen_txt, preproc_dir):
        save_conll(sen, f"{preproc_dir}/sen{sen_idx}.conll")
        parsed_doc = parse_doc(self.nlp, sen, preproc_dir, save=sen_txt != last_sen_txt)
        triplet = get_triplet(sen)
        triplet.to_file(f"{preproc_dir}/sen{sen_idx}_triplet.txt")
        ud_graph = get_ud_graph(parsed_doc)

        if sen_txt != last_sen_txt:
            json.dump(
                [n for n in nx.topological_sort(ud_graph.G)],
                open(f"{preproc_dir}/pos_edge_graph_top_order.json", "w")
            )
            bolinas_graph = ud_graph.pos_edge_graph(self.vocab)
            save_bolinas_str(f"{preproc_dir}/pos_edge.graph", bolinas_graph)
            save_bolinas_str(f"{preproc_dir}/pos_edge_with_labels.graph", bolinas_graph, add_names=True)
            add_node_labels(bolinas_graph)
            save_as_dot(f"{preproc_dir}/pos_edge_graph.dot", bolinas_graph)

        triplet_subgraph = get_triplet_subgraph(ud_graph, triplet, self.vocab)
        save_as_dot(f"{preproc_dir}/sen{sen_idx}_triplet_graph.dot", triplet_subgraph)
        save_bolinas_str(f"{preproc_dir}/sen{sen_idx}_triplet.graph", triplet_subgraph)

        add_triplet_data_to_node_name(ud_graph, triplet)
        save_as_dot(f"{preproc_dir}/sen{sen_idx}_ud.dot", ud_graph)

    def _after_loop(self):
        self.vocab.to_file(f"{self.vocab_file}")
        super()._after_loop()


if __name__ == "__main__":
    args = get_data_dir_and_config_args("Script to preprocess conll oie data.")
    script = PreprocScript(
        args.data_dir,
        args.config,
    )
    script.run()

