import json
from collections import defaultdict

import networkx as nx
import stanza
from stanza.utils.conll import CoNLL

from tuw_nlp.graph.ud_graph import UDGraph
from tuw_nlp.sem.hrg.common.io import save_bolinas_str, save_as_dot
from tuw_nlp.sem.hrg.common.script.loop_on_conll import LoopOnConll
from tuw_nlp.sem.hrg.common.triplet import Triplet


class Preproc(LoopOnConll):

    def __init__(self, config=None):
        super().__init__(description="Script to preprocess conll oie data.", config=config)

    def _before_loop(self):
        self.nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,mwt,pos,lemma,depparse",
            tokenize_pretokenized=True,
        )

    def _do_for_sen(self, sen_idx, sen, sen_txt, last_sen_txt, sen_dir):
        self._save_conll(sen, f"{sen_dir}/sen{sen_idx}.conll")
        parsed_doc = self._parse_doc(self.nlp, sen, sen_dir, save=sen_txt != last_sen_txt)
        triplet = self._get_triplet(sen)
        triplet.to_file(f"{sen_dir}/sen{sen_idx}_triplet.txt")
        ud_graph = UDGraph(parsed_doc.sentences[0])

        if sen_txt != last_sen_txt:
            save_as_dot(f"{sen_dir}/general_ud.dot", ud_graph)
            json.dump(
                [n for n in nx.topological_sort(ud_graph.G)],
                open(f"{sen_dir}/pos_edge_graph_top_order.json", "w")
            )
            bolinas_graph = ud_graph.pos_edge_graph()
            save_bolinas_str(f"{sen_dir}/pos_edge.graph", bolinas_graph)
            save_bolinas_str(f"{sen_dir}/pos_edge_with_labels.graph", bolinas_graph, add_names=True)
            self._add_node_labels(bolinas_graph)
            save_as_dot(f"{sen_dir}/pos_edge_graph.dot", bolinas_graph)

        triplet_ud = ud_graph.subgraph(list(triplet.node_to_label.keys()), handle_unconnected="shortest_path")
        triplet_pos_edge = triplet_ud.pos_edge_graph()
        save_bolinas_str(f"{sen_dir}/sen{sen_idx}_triplet.graph", triplet_pos_edge)
        self._add_node_labels(triplet_pos_edge)
        save_as_dot(f"{sen_dir}/sen{sen_idx}_triplet_graph.dot", triplet_pos_edge)

        self._add_triplet_data_to_node_name(triplet_ud, triplet)
        save_as_dot(f"{sen_dir}/sen{sen_idx}_ud.dot", triplet_ud)

    @staticmethod
    def _get_triplet(sen):
        triplet_dict = defaultdict(list)
        for i, tok in enumerate(sen):
            label = tok[7].split("-")[0]
            if label == "P" or label.startswith("A"):
                triplet_dict[label].append(i + 1)
        return Triplet(triplet_dict)

    @staticmethod
    def _add_triplet_data_to_node_name(graph, triplet, node_prefix=""):
        for n in graph.G.nodes:
            key = n
            if node_prefix:
                key = n.split(node_prefix)[1]
            label = triplet.get_label(key)
            if label is None:
                label = "X"
            new_name = graph.G.nodes[n]["name"]
            if new_name:
                new_name += "\n"
            new_name += f"{label}"
            graph.G.nodes[n]["name"] = new_name

    @staticmethod
    def _save_conll(sen, fn):
        with open(fn, 'w') as f:
            for line in sen:
                line[0] = str(int(line[0]) + 1)
                f.write("\t".join(line))
                f.write("\n")

    @staticmethod
    def _add_node_labels(bolinas_graph):
        for node, data in bolinas_graph.G.nodes(data=True):
            new_name = str(node)
            name = data['name']
            if name:
                new_name += f"\n{name}"
            data["name"] = new_name

    @staticmethod
    def _parse_doc(nlp, sen, out_dir, save=True):
        parsed_doc = nlp(" ".join(t[1] for t in sen))
        if save:
            fn = f"{out_dir}/parsed.conll"
            CoNLL.write_doc2conll(parsed_doc, fn)
        return parsed_doc


if __name__ == "__main__":
    Preproc().run()
