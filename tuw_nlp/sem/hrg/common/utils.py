import os.path
from collections import defaultdict

import networkx as nx
from stanza.utils.conll import CoNLL

from tuw_nlp.graph.graph import UnconnectedGraphError
from tuw_nlp.graph.ud_graph import UDGraph


def create_sen_dir(out_dir, sen_id):
    sen_dir = os.path.join(out_dir, str(sen_id))
    if not os.path.exists(sen_dir):
        os.makedirs(sen_dir)
    return sen_dir


def parse_doc(nlp, sen, sen_idx, out_dir, log):
    parsed_doc = nlp(" ".join(t[1] for t in sen))
    fn = f"{out_dir}/sen{sen_idx}_parsed.conll"
    CoNLL.write_doc2conll(parsed_doc, fn)
    log.write(f"wrote parse to {fn}\n")
    return parsed_doc


def get_ud_graph(parsed_doc):
    parsed_sen = parsed_doc.sentences[0]
    return UDGraph(parsed_sen)


def get_pred_and_args(sen, sen_idx, log):
    args = defaultdict(list)
    pred = []
    node_to_label = defaultdict()
    for i, tok in enumerate(sen):
        label = tok[7].split("-")[0]
        if label == "O":
            continue
        elif label == "P":
            pred.append(i + 1)
            node_to_label[i + 1] = label
            continue
        args[label].append(i + 1)
        node_to_label[i + 1] = label
    log.write(f"sen{sen_idx}\npred: {pred}\nargs: {args}\nnode_to_label: {node_to_label}\n")
    return args, pred, node_to_label


def save_bolinas_str(fn, graph, log, add_names=False):
    bolinas_graph = graph.to_bolinas(add_names=add_names)
    with open(fn, "w") as f:
        f.write(f"{bolinas_graph}\n")
    log.write(f"wrote graph to {fn}\n")


def save_as_dot(fn, graph, log):
    with open(fn, "w") as f:
        f.write(graph.to_dot())
    log.write(f"wrote graph to {fn}\n")


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


def add_labels_to_nodes(graph, gold_labels, pred_labels, node_prefix=""):
    for n in graph.G.nodes:
        gold, pred = "O", "O"
        key = n
        if node_prefix:
            key = n.split(node_prefix)[1]
        if int(key) < 1000:
            if key in gold_labels:
                gold = gold_labels[key]
            if key in pred_labels:
                pred = pred_labels[key]
            graph.G.nodes[n]["name"] = f"{gold}\n{pred}"


def resolve_pred(G, pred_labels, pos_tags, log=None):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    verbs = [n for n, t in pos_tags.items() if t == "VERB"]
    preds_w_verbs = [n for n in preds if n in verbs]
    if len(preds) == 1:
        if log:
            log.write(f"There is only one pred ({preds}), no resolution needed.\n")
        return
    if len(preds_w_verbs) == 1:
        keep = preds_w_verbs[0]
        for p in preds:
            if p != keep:
                del pred_labels[p]
        if log:
            log.write(f"Multiple preds ({preds}) but only one is verb ({preds_w_verbs}), only {keep} is kept.\n")
        return
    if len(preds_w_verbs) == 0 and len(preds) > 0:
        for p in preds:
            del pred_labels[p]
        if log:
            log.write(f"Multiple preds ({preds}) none of them is verb, all set back.\n")
    top_order = [n for n in nx.topological_sort(G)]
    if len(verbs) == 0:
        pred_labels[top_order[1].split('n')[1]] = "P"
        if log:
            log.write(f"There is no verb ({verbs}), root ({top_order[1].split('n')[1]}) is set to P.\n")
        return
    if len(verbs) == 1:
        pred_labels[verbs[0]] = "P"
        if log:
            log.write(f"There is only one verb ({verbs}), this one is set to P.\n")
        return
    assert len(verbs) > 1
    if len(preds_w_verbs) > 1:
        if log:
            log.write(f"Multiple verbs ({verbs}) and multiple preds with verbs ({preds_w_verbs}).\n")
        verbs = preds_w_verbs
    first_verb_idx = None
    for v_idx in verbs:
        idx = top_order.index(f"n{v_idx}")
        if first_verb_idx is None or idx < first_verb_idx:
            first_verb_idx = idx
    first_verb_node = top_order[first_verb_idx].split("n")[1]
    pred_labels[first_verb_node] = "P"
    if log:
        log.write(f"Multiple verbs ({verbs}), top one ({first_verb_node}) is set to P.\n")
    if len(preds_w_verbs) > 1:
        for p in preds:
            if p != first_verb_node:
                del pred_labels[p]
        if log:
            log.write(f"All P-s except for top ({first_verb_node}) are set back.\n")
    return


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


def add_arg_idx(extracted_labels, len):
    prev = "O"
    idx = -1
    for i in range(1, len+1):
        if str(i) not in extracted_labels:
            extracted_labels[str(i)] = "O"
        else:
            if extracted_labels[str(i)] == "A":
                if not prev.startswith("A"):
                    idx += 1
                extracted_labels[str(i)] = "A" + str(idx)
        prev = extracted_labels[str(i)]