import argparse
import json
import logging
import os
from collections import defaultdict

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.predict import resolve_pred, add_arg_idx
from tuw_nlp.sem.hrg.common.conll import get_pos_tags
from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-s", "--sens-file", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    return parser.parse_args()


def main(in_dir, sens_fn, first, last):
    with open(sens_fn) as f:
        sens = f.readlines()
    sens = [sen.strip() for sen in sens]

    graphs_dir = os.path.join(in_dir, "graphs")
    parsed_dir = os.path.join(in_dir, "parsed")
    bolinas_dir = os.path.join(in_dir, "bolinas")

    extracted = defaultdict(list)
    out_fn = os.path.join(in_dir, "extracted_wire.json")

    for sen_id in get_range(bolinas_dir, first, last):
        print(f"\nProcessing sentence {sen_id}")
        fn = os.path.join(bolinas_dir, f"{sen_id}.txt")
        if not os.path.exists(fn):
            continue
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
    main(args.in_dir, args.sens_file, args.first, args.last)
    