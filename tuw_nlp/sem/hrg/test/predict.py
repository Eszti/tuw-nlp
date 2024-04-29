import argparse
import json
import logging
import os
from collections import defaultdict

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.utils import resolve_pred, get_pos_tags, add_arg_idx


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-s", "--sens-file", type=str)
    return parser.parse_args()


def get_wire_extraction(extracted_labels, sen):
    words = sen.split(" ")
    labels = defaultdict(list)
    for i, word in enumerate(words):
        word_id = i + 1
        if extracted_labels[str(word_id)] != "O":
            labels[extracted_labels[str(word_id)]].append(word)
    arg2_keys = sorted([k for k in labels.keys() if not (k == "P" or k == "O" or k == "A0")])
    return {
        "arg1": " ".join(labels["A0"]),
        "rel": " ".join(labels["P"]),
        "arg2+": [" ".join(labels[key]) for key in arg2_keys],
        "score": "1.0",
        "extractor": "PoC",
    }


def main(in_dir, sens_fn):
    with open(sens_fn) as f:
        sens = f.readlines()
    sens = [sen.strip() for sen in sens]

    graphs_dir = os.path.join(in_dir, "graphs")
    parsed_dir = os.path.join(in_dir, "parsed")
    bolinas_dir = os.path.join(in_dir, "bolinas")

    extracted = defaultdict(list)
    out_fn = os.path.join(in_dir, "extracted_wire.json")

    for fn in sorted(os.listdir(bolinas_dir)):
        print(f"\nProcessing sentence {fn}")
        sen_id = int(fn.split(".")[0])
        sen = sens[sen_id]

        graph_file = f"{graphs_dir}/{sen_id}.graph"
        with open(os.path.join(graphs_dir, graph_file)) as f:
            lines = f.readlines()
            assert len(lines) == 1
            graph_str = lines[0].strip()

        graph = Graph.from_bolinas(graph_str)

        with open(os.path.join(bolinas_dir, fn)) as f:
            labels_lines = f.readlines()

        parsed_doc_file = f"{parsed_dir}/{sen_id}.conll"
        pos_tags = get_pos_tags(os.path.join(parsed_dir, parsed_doc_file))

        for labels_str in labels_lines:
            if labels_str.strip() in ["max"]:
                continue

            extracted_labels = json.loads(labels_str)
            resolve_pred(graph.G, extracted_labels, pos_tags)
            add_arg_idx(extracted_labels, len(pos_tags))
            extracted[sen].append(get_wire_extraction(extracted_labels, sen))

        with open(out_fn, "w") as f:
            json.dump(extracted, f, indent=4)


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)

    args = get_args()
    main(args.in_dir, args.sens_file)