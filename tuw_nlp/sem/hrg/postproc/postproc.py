import copy
import itertools
from collections import defaultdict


def resolve_pred(pred_labels, pos_tags, top_order):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    if len(preds) > 0:
        return
    verbs = [n for n, t in pos_tags.items() if t == "VERB"]
    if len(verbs) == 0:
        pred_labels[str(top_order[1])] = "P"
        return
    if len(verbs) == 1:
        pred_labels[verbs[0]] = "P"
        return
    assert len(verbs) > 1
    first_verb_idx = None
    for v_idx in verbs:
        idx = top_order.index(int(v_idx))
        if first_verb_idx is None or idx < first_verb_idx:
            first_verb_idx = idx
    first_verb_node = str(top_order[first_verb_idx])
    pred_labels[first_verb_node] = "P"
    return


def add_arg_idx(extracted_labels, length, arg_perm):
    prev = "O"
    idx = -1
    groups = defaultdict(list)
    for i in range(1, length + 1):
        if str(i) not in extracted_labels:
            extracted_labels[str(i)] = "O"
        else:
            if extracted_labels[str(i)] == "A":
                if not prev.startswith("A"):
                    idx += 1
                groups[idx].append(i)
                extracted_labels[str(i)] = "A" + str(idx)
        prev = extracted_labels[str(i)]
    if not arg_perm:
        return [extracted_labels]
    ret = []
    group_permutations = list(itertools.permutations(groups.keys()))
    for permutation in group_permutations:
        new_extractions = copy.copy(extracted_labels)
        for i, group_idx in enumerate(permutation):
            for word_idx in groups[group_idx]:
                new_extractions[str(word_idx)] = "A" + str(i)
        ret.append(new_extractions)
    return ret


def postprocess(extracted_labels, pos_tags, top_order, arg_perm):
    resolve_pred(extracted_labels, pos_tags, top_order)
    return add_arg_idx(extracted_labels, len(pos_tags), arg_perm)

