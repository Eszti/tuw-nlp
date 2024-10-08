import os

from stanza.utils.conll import CoNLL


def create_sen_dir(out_dir, sen_id):
    sen_dir = os.path.join(out_dir, str(sen_id))
    if not os.path.exists(sen_dir):
        os.makedirs(sen_dir)
    return sen_dir


def parse_doc(nlp, sen, out_dir, log, save=True):
    parsed_doc = nlp(" ".join(t[1] for t in sen))
    if save:
        fn = f"{out_dir}/parsed.conll"
        CoNLL.write_doc2conll(parsed_doc, fn)
        log.write(f"wrote parse to {fn}\n")
    return parsed_doc


def save_bolinas_str(fn, graph, log, add_names=False):
    bolinas_graph = graph.to_bolinas(add_names=add_names)
    with open(fn, "w") as f:
        f.write(f"{bolinas_graph}\n")
    log.write(f"wrote graph to {fn}\n")


def save_as_dot(fn, graph, log):
    with open(fn, "w") as f:
        f.write(graph.to_dot())
    log.write(f"wrote graph to {fn}\n")


def get_range(in_dir, first=None, last=None):
    sen_dirs = sorted([int(fn.split(".")[0]) for fn in os.listdir(in_dir)])
    if first is None or first < sen_dirs[0]:
        first = sen_dirs[0]
    if last is None or last > sen_dirs[-1]:
        last = sen_dirs[-1]
    return [n for n in sen_dirs if first <= n <= last]


def get_k_files_or_assert_all(files):
    k_files = [i for i in files if i.split("_")[-1].startswith("k")]
    if k_files:
        files = sorted(k_files, key=lambda x: int(x.split('.')[0].split("_")[-1].split("k")[-1]))
    else:
        assert len(files) == 1 and files[0].endswith("_all.json")
    return files
