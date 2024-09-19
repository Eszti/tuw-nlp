import argparse
import json
import logging
import os
from collections import defaultdict

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.conll import get_pos_tags, get_sen_text_from_conll_file
from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction
from tuw_nlp.sem.hrg.predict.postprocess import resolve_pred, add_arg_idx
from tuw_nlp.sem.hrg.predict.utils import save_predicted_conll, update_graph_labels, get_marked_nodes


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    return parser.parse_args()


def get_preproc_input(preproc_dir_root, sen_dir):
    preproc_dir = os.path.join(preproc_dir_root, sen_dir, "preproc")

    graph_file = f"{preproc_dir}/sen{sen_dir}_labels.graph"
    with open(graph_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        graph_str = lines[0].strip()
    graph = Graph.from_bolinas(graph_str)

    pa_graph_file = f"{preproc_dir}/sen{sen_dir}_pa.graph"
    with open(pa_graph_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        pa_graph_str = lines[0].strip()
    pa_graph = Graph.from_bolinas(pa_graph_str)

    node_to_label_file = f"{preproc_dir}/sen{sen_dir}_node_to_label.json"
    with open(node_to_label_file) as f:
        gold_labels = json.load(f)

    parsed_doc_file = f"{preproc_dir}/sen{sen_dir}_parsed.conll"
    pos_tags = get_pos_tags(parsed_doc_file)

    orig_conll = f"{preproc_dir}/sen{sen_dir}.conll"
    sen_text = get_sen_text_from_conll_file(orig_conll)
    return sen_text, graph, pa_graph, gold_labels, pos_tags, orig_conll


def get_bolinas_input(predict_dir_root, sen_dir, chart_filter):
    bolinas_dir = os.path.join(predict_dir_root, sen_dir, "bolinas", chart_filter)
    matches_file = f"{bolinas_dir}/sen{sen_dir}_matches.graph"
    labels_file = f"{bolinas_dir}/sen{sen_dir}_predicted_labels.txt"
    if not os.path.exists(matches_file):
        return []
    with open(matches_file) as f:
        matches_lines = f.readlines()
    with open(labels_file) as f:
        labels_lines = f.readlines()
    return zip(matches_lines, labels_lines)


def postprocess(extracted_labels, graph, pos_tags, postprocess_p):
    resolve_pred(graph.G, extracted_labels, pos_tags, postprocess_p)
    add_arg_idx(extracted_labels, len(pos_tags))


def main(data_dir, config_json, first, last):
    config = json.load(open(config_json))
    for name, c in config.items():
        preproc_dir_root = f"{data_dir}/{c['preproc_dir']}"
        predict_dir_root = f"{data_dir}/{c['predict_dir_root']}"
        for sen_dir in get_range(preproc_dir_root, first, last):
            sen_dir = str(sen_dir)
            print(f"\nProcessing sentence {sen_dir}")

            sen_text, graph, pa_graph, gold_labels, pos_tags, orig_conll = get_preproc_input(
                preproc_dir_root,
                sen_dir
            )

            predict_dir = f"{predict_dir_root}/{sen_dir}/predict"

            for chart_filter in c["bolinas_chart_filters"]:
                chart_filter_dir = f"{predict_dir}/{chart_filter}"
                if not os.path.exists(chart_filter_dir):
                    os.makedirs(chart_filter_dir)
                bolinas_matches = get_bolinas_input(predict_dir_root, sen_dir, chart_filter)

                wire_extractions = defaultdict(lambda: defaultdict(list))
                for i, (match_line, labels_str) in enumerate(bolinas_matches):
                    k = i + 1
                    match_str = match_line.split(';')[0]
                    score = match_line.split(';')[1].strip()
                    match_graph = Graph.from_bolinas(match_str)
                    with open(f"{chart_filter_dir}/sen{sen_dir}_match_{i}.dot", "w") as f:
                        f.write(match_graph.to_dot())

                    for subdir, pp_c in c["postprocess"].items():
                        extracted_labels = json.loads(labels_str)
                        pp_dir = f"{chart_filter_dir}/{subdir}"
                        if not os.path.exists(pp_dir):
                            os.makedirs(pp_dir)
                        postprocess(extracted_labels, graph, pos_tags, pp_c["postprocess_p"])
                        save_predicted_conll(
                            orig_conll,
                            extracted_labels,
                            f"{pp_dir}/sen{sen_dir}_extracted_k{i}.conll"
                        )
                        wire_extractions[subdir][sen_text].append(get_wire_extraction(
                            extracted_labels,
                            sen_text,
                            sen_id=int(sen_dir),
                            k=k,
                            score=score,
                            extractor=predict_dir_root.split("/")[-1].split("_")[1],
                        ))

                        update_graph_labels(
                            graph,
                            gold_labels,
                            extracted_labels,
                            node_prefix="n"
                        )
                        with open(f"{pp_dir}/sen{sen_dir}_match_{i}_graph.dot", "w") as f:
                            f.write(graph.to_dot(
                                marked_nodes=get_marked_nodes(graph, extracted_labels))
                            )
                for subdir, extractions in wire_extractions.items():
                    wire_json = f"{chart_filter_dir}/{subdir}/sen{sen_dir}_wire.json"
                    with open(wire_json, "w") as f:
                        json.dump(extractions, f, indent=4)


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)

    args = get_args()
    main(args.data_dir, args.config, args.first, args.last)
