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
    predicates = []
    for node, label in extracted_labels.items():
        if label == "P":
            predicates += [node]
    return f"[{' '.join([p for p in predicates])}]"


def update_graph_labels(graph, gold_labels, pred_labels, node_prefix=""):
    for n in graph.G.nodes:
        gold, pred = "O", "O"
        key = n
        if node_prefix:
            key = n.split(node_prefix)[1]
        if int(key) == 0:
            graph.G.nodes[n]["name"] = "Gold\nPred"
        elif int(key) < 1000:
            if key in gold_labels:
                gold = gold_labels[key]
            if key in pred_labels:
                pred = pred_labels[key]
            graph.G.nodes[n]["name"] = f"{gold}\n{pred}"


def get_marked_nodes(graph, extracted_labels):
    labeled_nodes = [f"n{k}" for k in extracted_labels.keys() if extracted_labels[k] != "O"]
    word_nodes = []
    for l in labeled_nodes:
        technical_node = [n for n in graph.G.successors(l) if int(n.split("n")[1]) >= 1000]
        assert len(technical_node) == 1
        word_nodes += technical_node
    return labeled_nodes + word_nodes
