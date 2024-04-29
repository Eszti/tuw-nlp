import argparse
import json
import logging
import os

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.utils import add_labels_to_nodes, resolve_pred, get_pos_tags, add_arg_idx


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


def save_predicted_conll(orig_conll, extracted_labels, extracted_conll):
    output = []
    with open(orig_conll) as f:
        lines = f.readlines()
    predicate = get_predicate(extracted_labels)
    for line in lines:
        line = line.strip()
        fields = line.split("\t")
        output.append(fields[:2] + [predicate, extracted_labels.get(fields[0], "O"), str(1.0)])
    with open(extracted_conll, "w") as f:
        for line in output:
            f.write("\t".join(line))
            f.write("\n")


def get_predicate(extracted_labels):
    predicate = None
    for node, label in extracted_labels.items():
        if label == "P":
            assert predicate is None
            predicate = node
    return predicate


def main(in_dir, first, last):
    for sen_dir in get_range(in_dir, first, last):
        print(f"\nProcessing sentence {sen_dir}")

        sen_dir = str(sen_dir)
        preproc_dir = os.path.join(in_dir, sen_dir, "preproc")
        bolinas_dir = os.path.join(in_dir, sen_dir, "bolinas")
        predict_dir = os.path.join(in_dir, sen_dir, "predict")
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
        match_dir = os.path.join(predict_dir, "matches")
        if not os.path.exists(match_dir):
            os.makedirs(match_dir)

        graph_file = f"{preproc_dir}/sen{sen_dir}_labels.graph"
        pa_graph_file = f"{preproc_dir}/sen{sen_dir}_pa.graph"
        node_to_label_file = f"{preproc_dir}/sen{sen_dir}_node_to_label.json"
        parsed_doc_file = f"{preproc_dir}/sen{sen_dir}_parsed.conll"
        orig_conll = f"{preproc_dir}/sen{sen_dir}.conll"

        matches_file = f"{bolinas_dir}/sen{sen_dir}_matches.graph"
        labels_file = f"{bolinas_dir}/sen{sen_dir}_predicted_labels.txt"

        log = open(f"{predict_dir}/sen{sen_dir}_pred.log", "w")
        extracted_conll = f"{predict_dir}/sen{sen_dir}_extracted.conll"

        with open(os.path.join(in_dir, sen_dir, graph_file)) as f:
            lines = f.readlines()
            assert len(lines) == 1
            graph_str = lines[0].strip()

        graph = Graph.from_bolinas(graph_str)

        with open(os.path.join(in_dir, sen_dir, node_to_label_file)) as f:
            gold_labels = json.load(f)

        with open(os.path.join(in_dir, sen_dir, pa_graph_file)) as f:
            lines = f.readlines()
            assert len(lines) == 1
            pa_graph_str = lines[0].strip()

        pa_graph = Graph.from_bolinas(pa_graph_str)

        with open(os.path.join(in_dir, sen_dir, matches_file)) as f:
            matches_lines = f.readlines()
        with open(os.path.join(in_dir, sen_dir, labels_file)) as f:
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

            extracted_labels = json.loads(labels_str)
            pos_tags = get_pos_tags(os.path.join(in_dir, sen_dir, parsed_doc_file))
            log.write(f"pred_labels (before): {extracted_labels}\n")
            resolve_pred(graph.G, extracted_labels, pos_tags, log)
            log.write(f"pred_labels (after): {extracted_labels}\n")
            add_arg_idx(extracted_labels, len(pos_tags))
            log.write(f"added arg indexes: {extracted_labels}\n")

            add_labels_to_nodes(graph, gold_labels, extracted_labels, node_prefix="n")

            match_graph = Graph.from_bolinas(match_str)

            with open(f"{match_dir}/sen{sen_dir}_match_{state}_{i}.dot", "w") as f:
                f.write(match_graph.to_dot())

            match_graph_nodes = set([n for n in match_graph.G.nodes])
            match_graph_edges = set([(u, v, d["color"]) for (u, v, d) in match_graph.G.edges(data=True)])
            pa_graph_nodes = set([n for n in pa_graph.G.nodes])
            pa_graph_edges = set([(u, v, d["color"]) for (u, v, d) in pa_graph.G.edges(data=True)])
            save_predicted_conll(orig_conll, extracted_labels, extracted_conll)

            print(f"Match {state} {i}")
            print(f"Node matches: {len(match_graph_nodes & pa_graph_nodes)}/{len(pa_graph_nodes)}")
            print(f"Edge matches: {len(match_graph_edges & pa_graph_edges)}/{len(pa_graph_edges)}")

            with open(f"{match_dir}/sen{sen_dir}_match_{state}_{i}_graph.dot", "w") as f:
                f.write(graph.to_dot(marked_nodes=match_graph_nodes))

            i += 1


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)

    args = get_args()
    main(args.in_dir, args.first, args.last)