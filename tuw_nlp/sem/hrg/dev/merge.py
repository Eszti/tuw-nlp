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
    parser.add_argument("-a", "--all", action="store_true")
    return parser.parse_args()


def main(in_dir, out_fn, first, last, k, all_k):
    k_min, k_max = 10, 10
    if k:
        k_min = k_max = k
    if all_k:
        k_min = 1
    k_list = range(k_min, k_max+1)

    all_ex_set = defaultdict(lambda: defaultdict(set))
    ex_stat = defaultdict(lambda: Counter())

    for sen_dir in get_range(in_dir, first, last):
        print(f"\nProcessing sentence {sen_dir}")

        sen_dir = str(sen_dir)
        predict_dir = os.path.join(in_dir, sen_dir, "predict")
        wire_json = f"{predict_dir}/sen{sen_dir}_wire.json"

        if not os.path.exists(wire_json):
            for ki in k_list:
                ex_stat[ki][0] += 1
            continue
        with open(wire_json) as f:
            extractions = json.load(f)
        assert len(extractions.keys()) == 1
        sen = list(extractions.keys())[0]
        all_extractions = extractions[sen]

        all_extractions.sort(key=lambda x: x["k"])
        top_k_extractions = all_extractions[:k_max]

        for i, ex in enumerate(top_k_extractions):
            wire_ex = WiReEx(ex)
            assert i+1 == wire_ex["k"]
            for j in range(i+1, k_max+1):
                if j >= k_min:
                    all_ex_set[j][sen].add(wire_ex)
        for ki in k_list:
            ex_stat[ki][len(all_ex_set[ki][sen])] += 1

    for ki, d in all_ex_set.items():
        all_ex_list = {}
        for sen, items in d.items():
            for item in items:
                item["extractor"] = item["extractor"].split("_")[0] + f"_k{ki}"
            all_ex_list[sen] = list(items)

        if all_k or k:
            out_fn_k = f"{out_fn.split('.')[0]}_k{ki}.json"
        else:
            out_fn_k = f"{out_fn.split('.')[0]}_all.json"
        with open(out_fn_k, "w") as f:
            json.dump(all_ex_list, f, indent=4)

        if not k and not all_k:
            print(f"\nKeeping all extractions.")
        else:
            print(f"\nKeeping top {ki} extractions.")
        sum = 0
        ex_stat_i = {key: v for key, v in sorted(ex_stat[ki].items())}
        for j, v in ex_stat_i.items():
            print(f"{j} extraction(s): {v} sen")
            sum += v
        print(f"Sum {sum} sens")
        print(f"Output saved to {out_fn_k}\n")


if __name__ == "__main__":
    args = get_args()
    main(args.in_dir, args.out_file, args.first, args.last, args.k, args.all)
