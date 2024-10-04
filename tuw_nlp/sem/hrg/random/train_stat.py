import argparse
import json
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.conll import get_labels_str
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--train-fn", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    parser.add_argument("-m", "--models", nargs="+", type=str)
    return parser.parse_args()


def main(train_fn, out_dir, models):
    seq_stat = defaultdict(lambda: defaultdict(list))
    nr_ex_stat = Counter()
    last_sen_txt = ""
    cnt = 1
    with open(train_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            for m in models:
                labels_seq = get_labels_str(sen, m)
                seq_stat[m][len(sen)].append(labels_seq)
            sen_txt = " ".join([word[1] for word in sen])
            if last_sen_txt == sen_txt:
                cnt += 1
            elif last_sen_txt != "":
                nr_ex_stat[cnt] += 1
                cnt = 1
            last_sen_txt = sen_txt
    for m, stat in seq_stat.items():
        print(f"stat len: {len(stat)}")
        print(f"{sorted(stat)}")
        with open(f"{out_dir}/train_seq_dist_{m}.json", "w") as f:
            json.dump({key: v for key, v in sorted(stat.items())}, f)
    with open(f"{out_dir}/train_nr_ex.json", "w") as f:
        json.dump({key: v for key, v in sorted(nr_ex_stat.items())}, f)


if __name__ == "__main__":
    args = get_args()
    main(args.train_fn, args.out_dir, args.models)
