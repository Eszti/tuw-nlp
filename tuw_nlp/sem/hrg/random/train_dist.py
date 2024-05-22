import argparse
import json
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.conll import get_labels_str
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--train-fn", type=str)
    parser.add_argument("-o1", "--out-seq", type=str)
    parser.add_argument("-o2", "--out-nr-ex", type=str)
    return parser.parse_args()


def main(train_fn, out_seq, out_nr_ex):
    seq_stat = defaultdict(list)
    nr_ex_stat = Counter()
    last_sen_txt = ""
    cnt = 1
    with open(train_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            print(f"Processing sen {sen_idx}")
            labels_seq = get_labels_str(sen)
            seq_stat[len(sen)].append(labels_seq)
            sen_txt = " ".join([word[1] for word in sen])
            if last_sen_txt == sen_txt:
                cnt += 1
            elif last_sen_txt != "":
                nr_ex_stat[cnt] += 1
                cnt = 1
            last_sen_txt = sen_txt
    print(f"stat len: {len(seq_stat)}")
    print(f"{sorted(seq_stat)}")
    with open(out_seq, "w") as f:
        json.dump(seq_stat, f)
    with open(out_nr_ex, "w") as f:
        json.dump(nr_ex_stat, f)


if __name__ == "__main__":
    args = get_args()
    main(args.train_fn, args.out_seq, args.out_nr_ex)
