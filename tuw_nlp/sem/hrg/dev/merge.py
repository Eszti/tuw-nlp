import argparse
import json
import os
from collections import defaultdict

from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.wire_extraction import WiReEx


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    return parser.parse_args()


def main(in_dir, first, last):
    all_ex_set = defaultdict(set)
    out_fn = "../data/extracted_poc_dev.json"

    for sen_dir in get_range(in_dir, first, last):
        print(f"\nProcessing sentence {sen_dir}")

        sen_dir = str(sen_dir)
        predict_dir = os.path.join(in_dir, sen_dir, "predict")
        wire_json = f"{predict_dir}/sen{sen_dir}_wire.json"
        if not os.path.exists(wire_json):
            continue
        with open(wire_json) as f:
            extractions = json.load(f)
        assert len(extractions.keys()) == 1
        sen = list(extractions.keys())[0]
        assert len(extractions[sen]) == 1
        for ex in extractions[sen]:
            wire_ex = WiReEx(ex)
            all_ex_set[sen].add(wire_ex)

    all_ex_list = {}
    for sen, items in all_ex_set.items():
        all_ex_list[sen] = list(items)
    with open(out_fn, "w") as f:
        json.dump(all_ex_list, f, indent=4)
    print(f"Output saved to {out_fn}")


if __name__ == "__main__":
    args = get_args()
    main(args.in_dir, args.first, args.last)
