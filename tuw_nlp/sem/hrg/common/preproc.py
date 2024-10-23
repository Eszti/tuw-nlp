import json
import os
from collections import defaultdict

from tuw_nlp.graph.graph import UnconnectedGraphError
from tuw_nlp.graph.ud_graph import UDGraph


def get_ud_graph(parsed_doc):
    parsed_sen = parsed_doc.sentences[0]
    return UDGraph(parsed_sen)


def get_pred_and_args(sen, sen_idx, log):
    args = defaultdict(list)
    pred = []
    gold_labels = defaultdict()
    for i, tok in enumerate(sen):
        label = tok[7].split("-")[0]
        if label == "O":
            continue
        elif label == "P":
            pred.append(i + 1)
            gold_labels[i + 1] = label
            continue
        args[label].append(i + 1)
        gold_labels[i + 1] = label
    log.write(f"sen{sen_idx}\npred: {pred}\nargs: {args}\nnode_to_label: {gold_labels}\n")
    return args, pred, gold_labels


def get_pred_arg_subgraph(ud_graph, pred, args, vocab, log):
    idx_to_keep = [n for nodes in args.values() for n in nodes] + pred
    log.write(f"idx_to_keep: {idx_to_keep}\n")
    return ud_graph.subgraph(idx_to_keep, handle_unconnected="shortest_path").pos_edge_graph(vocab)


def check_args(args, log, sen_idx, ud_graph, vocab):
    agraphs = {}
    all_args_connected = True
    for arg, nodes in args.items():
        try:
            agraph_ud = ud_graph.subgraph(nodes)
        except UnconnectedGraphError:
            log.write(
                f"unconnected argument ({nodes}) in sentence {sen_idx}, skipping\n"
            )
            all_args_connected = False
            continue

        agraphs[arg] = agraph_ud.pos_edge_graph(vocab)
    return agraphs, all_args_connected


def add_oie_data_to_nodes(graph, node_to_label, node_prefix=""):
    for n in graph.G.nodes:
        key = n
        if node_prefix:
            key = n.split(node_prefix)[1]
        if key in node_to_label:
            new_name = graph.G.nodes[n]["name"]
            if new_name:
                new_name += "\n"
            new_name += f"{node_to_label[key]}"
            graph.G.nodes[n]["name"] = new_name


def get_gold_labels(preproc_dir, sen_idx):
    gold_labels = []
    preproc_path = f"{preproc_dir}/{sen_idx}/preproc"
    files = [fn for fn in os.listdir(preproc_path) if fn.endswith("_gold_labels.json")]
    for fn in files:
        with open(f"{preproc_path}/{fn}") as f:
            gold_labels.append(json.load(f))
    return gold_labels


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
