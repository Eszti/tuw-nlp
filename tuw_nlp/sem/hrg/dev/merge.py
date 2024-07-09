import argparse
import json
import os
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.wire_extraction import WiReEx


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-file", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    parser.add_argument("-k", type=int)
    return parser.parse_args()


def main(in_dir, out_fn, first, last, k):
    all_ex_set = defaultdict(set)

    ex_stat = Counter()
    i = 0
    for sen_dir in get_range(in_dir, first, last):
        print(f"\nProcessing sentence {sen_dir}")

        sen_dir = str(sen_dir)
        predict_dir = os.path.join(in_dir, sen_dir, "predict")
        wire_json = f"{predict_dir}/sen{sen_dir}_wire.json"
        if not os.path.exists(wire_json):
            ex_stat[0] += 1
            continue
        with open(wire_json) as f:
            extractions = json.load(f)
        assert len(extractions.keys()) == 1
        sen = list(extractions.keys())[0]
        top_k = extractions[sen]
        if k:
            top_k = top_k[:k]
        for ex in top_k:
            wire_ex = WiReEx(ex)
            all_ex_set[sen].add(wire_ex)
        ex_stat[len(top_k)] += 1

    all_ex_list = {}
    for sen, items in all_ex_set.items():
        all_ex_list[sen] = list(items)
    with open(out_fn, "w") as f:
        json.dump(all_ex_list, f, indent=4)

    if k:
        print(f"\nKeeping top {k} extractions.")
    else:
        print(f"\nKeeping all extractions.")
    sum = 0
    for k, v in ex_stat.items():
        print(f"{k} extraction(s): {v} sen")
        sum += v
    print(f"Sum {sum} sens\n")

    print(f"Output saved to {out_fn}")


if __name__ == "__main__":
    args = get_args()
    main(args.in_dir, args.out_file, args.first, args.last, args.k)
