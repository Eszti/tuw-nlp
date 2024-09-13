import argparse
import json

import numpy as np
import pandas as pd

from tuw_nlp.sem.hrg.common.report import save_bar_diagram
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold-fn", type=str)
    parser.add_argument("-p", "--pred-files", nargs="+", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    return parser.parse_args()


def main(gold_fn, pred_files, out_dir):
    sen_ids = [0]
    k_values = dict()

    last_sen_txt = ""
    cnt = 1
    k_values["gold"] = []
    with open(gold_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            if cnt == 1 and sen_idx > 1:
                sen_ids.append(sen_idx-1)
            sen_txt = " ".join([word[1] for word in sen])
            if last_sen_txt == sen_txt:
                cnt += 1
            elif last_sen_txt != "":
                k_values["gold"].append(cnt)
                cnt = 1
            last_sen_txt = sen_txt
        if cnt != 1:
            k_values["gold"].append(cnt)

    for fn in pred_files:
        model = fn.split("/")[-1].split(".")[0]
        k_values[model] = [0] * len(sen_ids)
        with open(fn) as f:
            extractions = json.load(f)
        for sen, k_extr in extractions.items():
            assert len(k_extr) > 0
            sen_id = int(k_extr[-1]["sen_id"])
            k_extr.sort(key=lambda x: x["k"])
            k = k_extr[-1]["k"]
            idx = sen_ids.index(sen_id)
            k_values[model][idx] = k

    k_values_df = pd.DataFrame(data=k_values, index=sen_ids)
    k_values_df.index.name = 'sen_id'

    k_hist = k_values_df.apply(lambda x: x.value_counts()).fillna(0).astype(int)
    k_hist.index.name = 'k'
    k_hist.to_csv(f"{out_dir}/k_hist.tsv", sep="\t")
    save_bar_diagram(k_hist,
                     "Number of sentences",
                     "Distribution of number of extractions",
                     f"{out_dir}/k_stat_bar.png")

    for key in k_values:
        if key != "gold":
            k_values_df[f"{key.split('_')[1]} - gold"] = k_values_df[key] - k_values_df["gold"]
    k_values_df.to_csv(f"{out_dir}/k_values.tsv", sep="\t")

    k_differences = k_values_df.iloc[:, -len(pred_files):]\
        .apply(lambda x: x.value_counts(normalize=True)).fillna(0).round(4)
    k_differences.to_csv(f"{out_dir}/k_differences.tsv", sep="\t")

    with open(f"{out_dir}/k_corr.txt", "w") as f:
        for model in sorted(k_values.keys()):
            if model != "gold":
                f.writelines(f"{model}\n")
                corr = np.corrcoef(k_values["gold"], k_values[model])
                f.writelines(f"values corr:\n{corr}\n")
                corr = np.corrcoef(k_hist["gold"], k_hist[model])
                f.writelines(f"distribution corr:\n{corr}\n")
                f.writelines(f"sum extractions: {k_hist[model].sum()}\n\n")


if __name__ == "__main__":
    args = get_args()
    main(args.gold_fn, args.pred_files, args.out_dir)
