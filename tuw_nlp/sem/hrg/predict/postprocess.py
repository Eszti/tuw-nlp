import networkx as nx


def resolve_pred(G, pred_labels, pos_tags, postprocess, pred_stat):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    if postprocess == "keep" and len(preds) > 0:
        pred_stat.append("X")
        return
    verbs = [n for n, t in pos_tags.items() if t == "VERB"]
    preds_w_verbs = [n for n in preds if n in verbs]
    if len(preds) == 1:
        assert postprocess != "keep"
        if preds == preds_w_verbs:
            pred_stat.append("X")
            return
        else:
            assert len(preds_w_verbs) == 0
            pred_stat.append("A")
    if len(preds_w_verbs) == 1:
        pred_stat.append("B")
        keep = preds_w_verbs[0]
        for p in preds:
            if p != keep:
                del pred_labels[p]
        return
    if len(preds_w_verbs) == 0:
        if len(preds) > 1:
            pred_stat.append("C")
        for p in preds:
            del pred_labels[p]
    top_order = [n for n in nx.topological_sort(G)]
    if len(verbs) == 0:
        pred_stat.append("D")
        pred_labels[top_order[1].split('n')[1]] = "P"
        return
    if len(verbs) == 1:
        pred_stat.append("E")
        pred_labels[verbs[0]] = "P"
        return
    assert len(verbs) > 1
    if len(preds_w_verbs) > 1:
        pred_stat.append("F")
        verbs = preds_w_verbs
    else:
        pred_stat.append("G")
    first_verb_idx = None
    for v_idx in verbs:
        idx = top_order.index(f"n{v_idx}")
        if first_verb_idx is None or idx < first_verb_idx:
            first_verb_idx = idx
    first_verb_node = top_order[first_verb_idx].split("n")[1]
    pred_labels[first_verb_node] = "P"
    if len(preds_w_verbs) > 1:
        for p in preds:
            if p != first_verb_node:
                del pred_labels[p]
    return


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
