import argparse
import json
import logging
import os
from collections import defaultdict

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.predict import add_labels_to_nodes, resolve_pred, add_arg_idx
from tuw_nlp.sem.hrg.common.conll import get_pos_tags, get_sen_from_conll_file
from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-pp", "--preproc-dir", type=str)
    parser.add_argument("-pd", "--predict-dir", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    return parser.parse_args()


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


def main(preproc_dir_root, predict_dir_root, first, last):
    for sen_dir in get_range(preproc_dir_root, first, last):
        print(f"\nProcessing sentence {sen_dir}")

        sen_dir = str(sen_dir)
        preproc_dir = os.path.join(preproc_dir_root, sen_dir, "preproc")
        bolinas_dir = os.path.join(predict_dir_root, sen_dir, "bolinas")
        predict_dir = os.path.join(predict_dir_root, sen_dir, "predict")
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
        if not os.path.exists(matches_file):
            continue

        log = open(f"{predict_dir}/sen{sen_dir}_pred.log", "w")
        wire_json = f"{predict_dir}/sen{sen_dir}_wire.json"

        with open(graph_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            graph_str = lines[0].strip()

        graph = Graph.from_bolinas(graph_str)

        with open(node_to_label_file) as f:
            gold_labels = json.load(f)

        with open(pa_graph_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            pa_graph_str = lines[0].strip()

        pa_graph = Graph.from_bolinas(pa_graph_str)

        with open(matches_file) as f:
            matches_lines = f.readlines()
        with open(labels_file) as f:
            labels_lines = f.readlines()
        state = None

        sen = get_sen_from_conll_file(orig_conll)
        wire_extractions = defaultdict(list)
        i = 1
        for (match_line, labels_str) in zip(matches_lines, labels_lines):
            if match_line.strip() in ["max", "prec", "rec"]:
                assert match_line.strip() == labels_str.strip()
                state = match_line.strip()
                i = 1
                log.write(f"{state}\n")
                continue
            log.write(f"k={i}\n")

            match_str = match_line.split(';')[0]
            score = match_line.split(';')[1].strip()

            extracted_labels = json.loads(labels_str)
            pos_tags = get_pos_tags(parsed_doc_file)
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
            extracted_conll = f"{predict_dir}/sen{sen_dir}_extracted_k{i}.conll"
            save_predicted_conll(orig_conll, extracted_labels, extracted_conll)
            if state == "max":
                wire_extractions[sen].append(get_wire_extraction(
                    extracted_labels,
                    sen,
                    sen_id=int(sen_dir),
                    k=i,
                    score=score
                ))

            print(f"Match {state} {i}")
            print(f"Node matches: {len(match_graph_nodes & pa_graph_nodes)}/{len(pa_graph_nodes)}")
            print(f"Edge matches: {len(match_graph_edges & pa_graph_edges)}/{len(pa_graph_edges)}")

            with open(f"{match_dir}/sen{sen_dir}_match_{state}_{i}_graph.dot", "w") as f:
                f.write(graph.to_dot(marked_nodes=match_graph_nodes))

            i += 1
        with open(wire_json, "w") as f:
            json.dump(wire_extractions, f, indent=4)


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)

    args = get_args()
    main(args.preproc_dir, args.predict_dir, args.first, args.last)