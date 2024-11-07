import argparse
import os

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


def get_range(in_dir, first=None, last=None):
    sen_dirs = sorted([int(fn.split(".")[0]) for fn in os.listdir(in_dir)])
    if first is None or first < sen_dirs[0]:
        first = sen_dirs[0]
    if last is None or last > sen_dirs[-1]:
        last = sen_dirs[-1]
    return [n for n in sen_dirs if first <= n <= last]


def log_to_console_and_log_lines(line, sen_log_lines):
    print(line)
    sen_log_lines.append(f"{line}\n")


def get_data_dir_and_config_args(desc=""):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    return parser.parse_args()
