import argparse
import json
import logging
import os

import networkx as nx

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.utils import add_labels_to_nodes


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    return parser.parse_args()


def get_range(in_dir, first, last):
    sen_dirs = sorted([int(d) for d in os.listdir(in_dir)])
    if first is None or first < sen_dirs[0]:
        first = sen_dirs[0]
    if last is None or last > sen_dirs[-1]:
        last = sen_dirs[-1]
    return range(first,  last + 1)


def get_pos_tags(fn):
    with open(fn) as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        line = line.strip()
        fields = line.split('\t')
        if len(fields) > 1:
            ret[fields[0]] = fields[3]
    return ret


def resolve_pred(G, pred_labels, pos_tags, log):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    verbs = [n for n, t in pos_tags.items() if t == "VERB"]
    preds_w_verbs = [n for n in preds if n in verbs]
    if len(preds) == 1:
        log.write(f"There is only one pred ({preds}), no resolution needed.\n")
        return
    if len(preds_w_verbs) == 1:
        keep = preds_w_verbs[0]
        for p in preds:
            if p != keep:
                del pred_labels[p]
        log.write(f"Multiple preds ({preds}) but only one is verb ({preds_w_verbs}), only {keep} is kept.\n")
        return
    if len(preds_w_verbs) == 0 and len(preds) > 0:
        for p in preds:
            del pred_labels[p]
        log.write(f"Multiple preds ({preds}) none of them is verb, all set back.\n")
    top_order = [n for n in nx.topological_sort(G)]
    if len(verbs) == 0:
        pred_labels[top_order[1].split('n')[1]] = "P"
        log.write(f"There is no verb ({verbs}), root ({top_order[1].split('n')[1]}) is set to P.\n")
        return
    if len(verbs) == 1:
        pred_labels[verbs[0]] = "P"
        log.write(f"There is only one verb ({verbs}), this one is set to P.\n")
        return
    assert len(verbs) > 1
    if len(preds_w_verbs) > 1:
        log.write(f"Multiple verbs ({verbs}) and multiple preds with verbs ({preds_w_verbs}).\n")
        verbs = preds_w_verbs
    first_verb_idx = None
    for v_idx in verbs:
        idx = top_order.index(f"n{v_idx}")
        if first_verb_idx is None or idx < first_verb_idx:
            first_verb_idx = idx
    first_verb_node = top_order[first_verb_idx].split("n")[1]
    pred_labels[first_verb_node] = "P"
    log.write(f"Multiple verbs ({verbs}), top one ({first_verb_node}) is set to P.\n")
    if len(preds_w_verbs) > 1:
        for p in preds:
            if p != first_verb_node:
                del pred_labels[p]
        log.write(f"All P-s except for top ({first_verb_node}) are set back.\n")
    return


def main(in_dir, first, last):
    for sen_dir in get_range(in_dir, first, last):
        print(f"\nProcessing sentence {sen_dir}")
        log = open(f"{in_dir}/{sen_dir}/sen{sen_dir}_pred.log", "w")

        sen_dir = str(sen_dir)
        out_dir = os.path.join(in_dir, sen_dir, "matches")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        files = os.listdir(os.path.join(in_dir, sen_dir))
        graph_file = [file for file in files if file.endswith(f"sen{sen_dir}_labels.graph")]
        assert len(graph_file) == 1
        pa_graph_file = [file for file in files if file.endswith("_pa.graph")]
        assert len(pa_graph_file) == 1
        matches_file = [file for file in files if file.endswith("_matches.graph")]
        assert len(matches_file) == 1
        labels_file = [file for file in files if file.endswith("_predicted_labels.txt")]
        assert len(labels_file) == 1
        node_to_label_file = [file for file in files if file.endswith("node_to_label.json")]
        assert len(node_to_label_file) == 1
        parsed_doc_file = [file for file in files if file.endswith("parsed.conll")]
        assert len(parsed_doc_file) == 1

        with open(os.path.join(in_dir, sen_dir, graph_file[0])) as f:
            lines = f.readlines()
            assert len(lines) == 1
            graph_str = lines[0].strip()

        graph = Graph.from_bolinas(graph_str)

        with open(os.path.join(in_dir, sen_dir, node_to_label_file[0])) as f:
            gold_labels = json.load(f)

        with open(os.path.join(in_dir, sen_dir, pa_graph_file[0])) as f:
            lines = f.readlines()
            assert len(lines) == 1
            pa_graph_str = lines[0].strip()

        pa_graph = Graph.from_bolinas(pa_graph_str)

        with open(os.path.join(in_dir, sen_dir, matches_file[0])) as f:
            matches_lines = f.readlines()
        with open(os.path.join(in_dir, sen_dir, labels_file[0])) as f:
            labels_lines = f.readlines()
        state = None
        i = 0
        for (match_str, labels_str) in zip(matches_lines, labels_lines):
            if match_str.strip() in ["max", "prec", "rec"]:
                assert match_str.strip() == labels_str.strip()
                state = match_str.strip()
                i = 0
                log.write(f"{state}\n")
                continue
            log.write(f"k={i}\n")
            
            pred_labels = json.loads(labels_str)
            pos_tags = get_pos_tags(os.path.join(in_dir, sen_dir, parsed_doc_file[0]))
            log.write(f"pred_labels (before): {pred_labels}\n")
            resolve_pred(graph.G, pred_labels, pos_tags, log)
            log.write(f"pred_labels (after): {pred_labels}\n")

            add_labels_to_nodes(graph, gold_labels, pred_labels, node_prefix="n")

            match_graph = Graph.from_bolinas(match_str)

            with open(f"{out_dir}/sen{sen_dir}_match_{state}_{i}.dot", "w") as f:
                f.write(match_graph.to_dot())

            match_graph_nodes = set([n for n in match_graph.G.nodes])
            match_graph_edges = set([(u, v, d["color"]) for (u, v, d) in match_graph.G.edges(data=True)])
            pa_graph_nodes = set([n for n in pa_graph.G.nodes])
            pa_graph_edges = set([(u, v, d["color"]) for (u, v, d) in pa_graph.G.edges(data=True)])

            print(f"Match {state} {i}")
            print(f"Node matches: {len(match_graph_nodes & pa_graph_nodes)}/{len(pa_graph_nodes)}")
            print(f"Edge matches: {len(match_graph_edges & pa_graph_edges)}/{len(pa_graph_edges)}")

            with open(f"{out_dir}/sen{sen_dir}_match_{state}_{i}_graph.dot", "w") as f:
                f.write(graph.to_dot(marked_nodes=match_graph_nodes))

            i += 1


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)

    args = get_args()
    main(args.in_dir, args.first, args.last)