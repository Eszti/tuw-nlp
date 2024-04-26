import argparse
import sys

import stanza

from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.sem.hrg.common.utils import get_ud_graph


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
        ud_graph = get_ud_graph(parsed_doc)
        pos_edge_graph = ud_graph.pos_edge_graph(vocab)
        bolinas_graph = pos_edge_graph.to_bolinas()
        fn = f"{out_dir}/{sen_idx}.graph"
        with open(fn, "w") as f:
            f.write(f"{bolinas_graph}\n")


if __name__ == "__main__":
    args = get_args()
    main(args.out_dir)
