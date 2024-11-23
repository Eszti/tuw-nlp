from tuw_nlp.sem.hrg.steps.bolinas.common.output import print_shifted, format_derivation


def get_labels(derivation):
    if type(derivation) is not tuple:
        if derivation == "START" or derivation.rule.symbol == "S":
            return {}
        return {derivation.mapping['_1'].split('n')[1]: derivation.rule.symbol}
    else:
        ret = {}
        items = [c for (_, c) in derivation[1].items()] + [derivation[0]]
        for item in items:
            for (k, v) in get_labels(item).items():
                assert k not in ret
                ret[k] = v
        return ret


def get_rules(derivation, grammar_lines):
    if type(derivation) is not tuple:
        if derivation == "START":
            return {}
        rule_id = derivation.rule.rule_id
        return {rule_id: grammar_lines[rule_id-1]}
    else:
        ret = {}
        items = [c for (_, c) in derivation[1].items()] + [derivation[0]]
        for item in items:
            for (k, v) in get_rules(item, grammar_lines).items():
                assert k not in ret or v == ret[k]
                ret[k] = v
        return ret


def extract_for_kth_derivation(derivation, n_score, ki, grammar_lines):
    shifted_derivation = f"%s;%g\n" % (print_shifted(derivation), n_score)
    used_rules = "%s\t#%g\n" % (format_derivation(derivation), n_score)
    rules = get_rules(derivation, grammar_lines)
    for rule_id in sorted(rules):
        rule_str = rules[rule_id]
        prob = rule_str.split(';')[1].strip()
        if not prob:
            prob = 0
        rule = rule_str.split(';')[0].strip()
        used_rules += "%s\t%.2f\t%s\n" % (rule_id, float(prob), rule)

    final_item = derivation[1]["START"][0]
    nodes = sorted(list(final_item.nodeset), key=lambda node: int(node[1:]))
    matched_nodes = "k%d:\t%s\n" % (ki, nodes)
    return shifted_derivation, used_rules, matched_nodes