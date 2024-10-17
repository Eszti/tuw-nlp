import copy
from collections import defaultdict

from tuw_nlp.sem.hrg.bolinas.common.oie import get_labels
from tuw_nlp.sem.hrg.postproc.postproc import postprocess


def f1(prec, rec):
    try:
        return 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        return 0


def convert_label_dict(labels):
    ret = defaultdict(list)
    for n, l in labels.items():
        ret[l].append(n)
    return ret


def match(gold, pred):
    p_match = set(gold["P"]) & set(pred["P"])
    a0_match = set(gold["A0"]) & set(pred["A0"])
    return p_match and a0_match


def calc_len(labels):
    ret = 0
    for label, nodes in labels.items():
        if label.startswith("P") or label.startswith("A"):
            ret += len(nodes)
    return ret


def calc_p_and_g(gold, pred):
    ret = 0
    for label, gold_nodes in gold.items():
        if label.startswith("P") or label.startswith("A"):
            pred_nodes = pred.get(label, [])
            ret += len(set(gold_nodes) & set(pred_nodes))
    return ret


def calculate_score(gold, pred, metric):
    p_len, g_len = 0.0, 0.0
    g_and_p = calc_p_and_g(gold, pred)
    if metric in ["prec", "f1"]:
        p_len = calc_len(pred)
        if metric == "prec":
            return g_and_p / float(p_len)
    if metric in ["rec", "f1"]:
        g_len = calc_len(gold)
        if metric == "rec":
            return g_and_p / float(g_len)
    assert metric == "f1"
    prec = g_and_p / float(p_len)
    rec = g_and_p / float(g_len)
    return f1(prec, rec)


def filter_for_pr(derivations, gold_labels, metric, pos_tags, top_order, arg_perm):
    derivations_to_keep = [None] * len(gold_labels)
    extractions = [None] * len(gold_labels)
    max_scores = [-1.0] * len(gold_labels)

    for score, derivation in derivations:
        labels = get_labels(derivation)
        predicted_labels = copy.copy(labels)
        _, all_permutations = postprocess(predicted_labels, pos_tags, top_order, arg_perm)

        for permutation in all_permutations:
            pred = convert_label_dict(permutation)
            for i, g in enumerate(gold_labels):
                gold = convert_label_dict(g)
                if not match(gold, pred):
                    continue
                score = calculate_score(gold, pred, metric)
                if score > max_scores[i]:
                    derivations_to_keep[i] = derivation
                    max_scores[i] = score
                    extractions[i] = {k: v for k, v in permutation.items() if v != "O"}

    found_derivations = []
    found_extractions = []
    scores = []
    for i, d in enumerate(derivations_to_keep):
        if d is None:
            assert max_scores[i] == -1.0
        else:
            found_derivations.append(d)
            found_extractions.append(extractions[i])
            scores.append(max_scores[i])
    if not scores:
        return [], []

    ret = zip(scores, found_derivations, found_extractions)
    ret = sorted(ret, reverse=True, key=lambda x: x[0])
    a, b, c = zip(*ret)
    d = list(zip(a, b))
    return d, c
