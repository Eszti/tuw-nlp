import argparse
import json
import os

import numpy as np
import pandas as pd

from tuw_nlp.sem.hrg.common.io import get_all_json
from tuw_nlp.sem.hrg.common.report import save_bar_diagram


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    return parser.parse_args()


def get_pred_files(data_dir, config):
    ret = []
    for c in config["models"]:
        if c.get("ignore") and c["ignore"]:
            continue
        for chart_filter in sorted(c["bolinas_chart_filters"]):
            for pp in sorted(c["postprocess"]):
                ret.append(get_all_json(f"{data_dir}/{config['extractions_dir']}/{c['in_dir']}", chart_filter, pp))
    return ret


def main(data_dir, config_json):
    config = json.load(open(config_json))
    out_dir = f"{os.path.dirname(os.path.realpath(__file__))}/reports/k_stat"
    gold_fn = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data/{config['gold_fn']}"
    sen_ids = []
    k_values = dict()

    k_values["gold"] = []
    gold_extractions = json.load(open(gold_fn))
    for sen, extractions in gold_extractions.items():
        sen_ids.append(int(sorted(extractions, key=lambda x: int(x["sen_id"]))[0]["sen_id"]))
        k_values["gold"].append(len(extractions))

    pred_files = get_pred_files(data_dir, config)
    for fn in pred_files:
        model = fn.split("/")[-1].split(".")[0].split("dev_")[-1].split("_all")[0]
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
            k_values_df[f"{key} - gold"] = k_values_df[key] - k_values_df["gold"]
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
    main(args.data_dir, args.config)
