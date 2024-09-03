import argparse
import json
import os
from collections import Counter, defaultdict

from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold-fn", type=str)
    parser.add_argument("-p", "--pred-dirs", nargs="+", type=str)
    parser.add_argument("-o", "--out-fn", type=str)
    return parser.parse_args()


def main(gold_fn, pred_dirs, out_fn):
    gold_stat = Counter()
    pred_stats = defaultdict(Counter)

    last_sen_txt = ""
    cnt = 1
    gold_stat[0] = 0
    with open(gold_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            sen_txt = " ".join([word[1] for word in sen])
            if last_sen_txt == sen_txt:
                cnt += 1
            elif last_sen_txt != "":
                gold_stat[cnt] += 1
                cnt = 1
            last_sen_txt = sen_txt

    for in_dir in pred_dirs:
        for sen_dir in get_range(in_dir):
            bolinas_dir = os.path.join(in_dir, str(sen_dir), "bolinas")
            matches_file = f"{bolinas_dir}/sen{sen_dir}_matches.graph"
            k = 0
            if os.path.exists(matches_file):
                with open(os.path.join(in_dir, str(sen_dir), matches_file)) as f:
                    matches_lines = f.readlines()
                for match_line in matches_lines:
                    if match_line.strip() in ["max", "prec", "rec"]:
                        continue
                    k += 1
            pred_stats[in_dir][k] += 1

    with open(out_fn, "w") as f:
        f.writelines("Gold stat:\n")
        json.dump({key: v for key, v in sorted(gold_stat.items())}, f)
        f.writelines("\n\n")
        for in_dir, pred_stat in pred_stats.items():
            f.writelines(f"{in_dir}\n")
            json.dump({key: v for key, v in sorted(pred_stat.items())}, f)
            f.writelines("\n\n")


if __name__ == "__main__":
    args = get_args()
    main(args.gold_fn, args.pred_dirs, args.out_fn)
