import networkx as nx


def resolve_pred(G, pred_labels, pos_tags, postprocess, log=None):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    if not postprocess and len(preds) > 0:
        return
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
