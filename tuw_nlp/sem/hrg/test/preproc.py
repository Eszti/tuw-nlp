import argparse
import os
import sys

import stanza
from stanza.utils.conll import CoNLL

from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.sem.hrg.common.preproc import get_ud_graph


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--out-dir", default="out", type=str)
    return parser.parse_args()


def main(out_dir):
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )
    vocab = Vocabulary(first_id=1000)
    for sen_idx, line in enumerate(sys.stdin):
        print(f"Processing sen {sen_idx}")
        sen = line.strip()
        parsed_doc = nlp(sen)

        parsed_dir = os.path.join(out_dir, "parsed")
        if not os.path.exists(parsed_dir):
            os.makedirs(parsed_dir)
        fn = f"{parsed_dir}/{sen_idx}.conll"
        CoNLL.write_doc2conll(parsed_doc, fn)

        ud_graph = get_ud_graph(parsed_doc)
        pos_edge_graph = ud_graph.pos_edge_graph(vocab)
        bolinas_graph = pos_edge_graph.to_bolinas()

        graphs_dir = os.path.join(out_dir, "graphs")
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)
        fn = f"{graphs_dir}/{sen_idx}.graph"
        with open(fn, "w") as f:
            f.write(f"{bolinas_graph}\n")


if __name__ == "__main__":
    args = get_args()
    main(args.out_dir)
