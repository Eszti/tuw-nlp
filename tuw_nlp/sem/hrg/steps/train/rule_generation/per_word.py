from collections import defaultdict

import networkx as nx


def get_next_edges(G, root_word, triplet, log):
    next_edges = defaultdict(list)
    root_pos = ""
    for _, v, e in G.edges(root_word, data=True):
        node_idx = int(v.split("n")[-1])
        label = triplet.get_label(node_idx)
        if label:
            next_edges[label[0]].append((e['color'], v))
        elif node_idx >= 1000:
            root_pos = e['color']
        else:
            next_edges['X'].append((e['color'], v))
    log.write(f"next_edges: {next_edges}\n")
    return next_edges, root_pos


def gen_subseq_rules(G, pred_edges, triplet, log):
    for lhs, edges in pred_edges.items():
        for dep_rel, node in edges:
            next_edges, root_pos = get_next_edges(G, node, triplet, log)

            rule = f'{lhs} -> (. :{dep_rel} (.'
            for non_term in next_edges:
                rule += ' ' + ' '.join(f':{non_term}$' for _ in next_edges[non_term])
            rule += f' :{root_pos} .));\n'
            yield rule

            yield from gen_subseq_rules(G, next_edges, triplet, log)


def get_initial_rule(next_edges, root_pos):
    rule = 'S -> (.'
    for lhs in sorted(next_edges.keys()):
        rule += ' ' + ' '.join(f':{lhs}$' for _ in next_edges[lhs])
    rule += f' :{root_pos} .);\n'
    return rule


def get_rules_per_word(triplet_graph, triplet, log):
    root_word = next(nx.topological_sort(triplet_graph.G))
    log.write(f"root word: {root_word}\n")
    next_edges, root_pos = get_next_edges(triplet_graph.G, root_word, triplet, log)
    initial_rule = get_initial_rule(next_edges, root_pos)
    rules = set()
    for rule in gen_subseq_rules(triplet_graph.G, next_edges, triplet, log):
        rules.add(rule)
    return initial_rule, rules
