import argparse
import json
import random
from collections import defaultdict

from tuw_nlp.sem.hrg.common.predict import add_arg_idx
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t1", "--train-seq-dist", type=str)
    parser.add_argument("-t2", "--train-nr-ex", type=str)
    parser.add_argument("-d", "--dev-fn", type=str)
    parser.add_argument("-o", "--out-fn", type=str)
    return parser.parse_args()


def main(train_seq_dist, train_nr_ex, dev_fn, out_fn):
    with open(train_seq_dist) as f:
        seq_stat = json.load(f)
    with open(train_nr_ex) as f:
        nr_ex_stat = json.load(f)
    print(f"stat len: {len(seq_stat)}")
    print(f"{sorted(seq_stat)}")
    extracted = defaultdict(list)
    random.seed(10)

    with open(dev_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            print(f"Processing sen {sen_idx}")
            sen_len = str(len(sen))
            nr_seq = len(seq_stat[sen_len])
            sen_txt = " ".join([line[1] for line in sen])
            rnd_idx = random.randrange(nr_seq)
            pred_seq = seq_stat[sen_len][rnd_idx]
            extracted_labels = {str(i+1): l for i, l in enumerate(pred_seq)}
            add_arg_idx(extracted_labels, len(pred_seq))
            extracted[sen_txt].append(get_wire_extraction(extracted_labels, sen_txt, extractor="rnd_complete_seq"))

    with open(out_fn,"w") as f:
        json.dump(extracted, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args.train_seq_dist, args.train_nr_ex, args.dev_fn, args.out_fn)
