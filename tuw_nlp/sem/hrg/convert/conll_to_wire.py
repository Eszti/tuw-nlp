import argparse
import json
import sys
from collections import defaultdict

from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction_from_conll
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--out-fn", type=str)
    return parser.parse_args()


def main(out_fn):
    wire_all = defaultdict(list)
    for sen_idx, sen in enumerate(gen_tsv_sens(sys.stdin)):
        print(f"Processing sen {sen_idx}")
        sen_txt = " ".join([fields[1] for fields in sen])
        wire_ex = get_wire_extraction_from_conll(sen)
        wire_all[sen_txt].append(wire_ex)
    with open(out_fn, "w") as f:
        json.dump(wire_all, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args.out_fn)