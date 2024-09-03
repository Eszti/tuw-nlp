import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.report import save_bar_diagram
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold-fn", type=str)
    parser.add_argument("-p", "--pred-dirs", nargs="+", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    return parser.parse_args()


def main(gold_fn, pred_dirs, out_dir):
    gold_hist = Counter()
    pred_hist = defaultdict(Counter)
    sen_ids = [0]
    k_values = defaultdict(list)

    last_sen_txt = ""
    cnt = 1
    gold_hist[0] = 0
    with open(gold_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            if cnt == 1 and sen_idx > 1:
                sen_ids.append(sen_idx-1)
            sen_txt = " ".join([word[1] for word in sen])
            if last_sen_txt == sen_txt:
                cnt += 1
            elif last_sen_txt != "":
                gold_hist[cnt] += 1
                k_values["gold"].append(cnt)
                cnt = 1
            last_sen_txt = sen_txt
        if cnt != 1:
            gold_hist[cnt] += 1
            k_values["gold"].append(cnt)

    for in_dir in pred_dirs:
        for i, sen_dir in enumerate(get_range(in_dir)):
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
            in_dir_str = in_dir.split("/")[-1]
            pred_hist[in_dir_str][k] += 1
            assert sen_ids[i] == sen_dir
            k_values[in_dir_str].append(k)

    df = pd.DataFrame(data=k_values, index=sen_ids)
    df.to_csv(f"{out_dir}/k_values.tsv", sep="\t")

    with open(f"{out_dir}/dev_k_stat.txt", "w") as f:
        f.writelines("Gold stat:\n")
        json.dump({key: v for key, v in sorted(gold_hist.items())}, f)
        f.writelines(f"\n{sum(gold_hist.values())}\n\n")
        for in_dir, pred_stat in pred_hist.items():
            f.writelines(f"{in_dir}\n")
            json.dump({key: v for key, v in sorted(pred_stat.items())}, f)
            corr = np.corrcoef(k_values["gold"], k_values[in_dir])
            f.writelines(f"\ncorr:\n{corr}\n{sum(pred_stat.values())}\n\n")

    labels = list(sorted(gold_hist.keys()))  # beware if k is greater in preds
    all_stat = dict()
    all_stat["gold"] = [gold_hist[l] for l in labels]
    for in_dir, pred_stat in pred_hist.items():
        all_stat[in_dir] = [pred_stat[l] for l in labels]
    save_bar_diagram(labels,
                     all_stat,
                     "Number of sentences",
                     "Distribution of number of extractions",
                     f"{out_dir}/dev_k_stat_bar.png")


if __name__ == "__main__":
    args = get_args()
    main(args.gold_fn, args.pred_dirs, args.out_dir)
