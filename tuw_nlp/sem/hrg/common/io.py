from stanza.utils.conll import CoNLL


def parse_doc(nlp, sen, out_dir, save=True):
    parsed_doc = nlp(" ".join(t[1] for t in sen))
    if save:
        fn = f"{out_dir}/parsed.conll"
        CoNLL.write_doc2conll(parsed_doc, fn)
    return parsed_doc


def save_bolinas_str(fn, graph, add_names=False):
    bolinas_graph = graph.to_bolinas(add_names=add_names)
    with open(fn, "w") as f:
        f.write(f"{bolinas_graph}\n")


def save_as_dot(fn, graph):
    with open(fn, "w") as f:
        f.write(graph.to_dot())


def log_to_console_and_log_lines(line, sen_log_lines):
    print(line)
    sen_log_lines.append(f"{line}\n")
