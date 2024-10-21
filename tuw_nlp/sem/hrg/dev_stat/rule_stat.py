import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from tuw_nlp.sem.hrg.common.io import get_range, get_data_dir_and_config_args


def get_rules_from_grammar_file(config, grammar_dir):
    grammar_fn = f"{grammar_dir}/{config['grammar_file']}"
    with open(grammar_fn) as f:
        lines = f.readlines()
        rules = [(
            l.split("\t")[0].split(";")[0],
            l.split("\t")[0].split(";")[0].split(" ")[0],
            float(l.split("\t")[1]),
        ) for l in lines]
    return rules


def fill_stat(bf, c, data_dir, sen_dir, stat_dict):
    derivation_file = f"{data_dir}/{c['in_dir']}/{sen_dir}/bolinas/{bf}/sen{sen_dir}_derivation.txt"
    if os.path.exists(derivation_file):
        with open(derivation_file) as f:
            lines = f.readlines()
            for line in lines:
                if "#" in line or not line or line == "\n":
                    continue
                rule_id = int(line.split("\t")[0])
                stat_dict[bf][rule_id-1] += 1


def calc_rule_stat(data_dir, config_json):
    grammar_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/train/grammar"
    report_dir = f"{os.path.dirname(os.path.realpath(__file__))}/reports/rule_stat"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    config = json.load(open(config_json))

    for c in config["models"]:
        rules = get_rules_from_grammar_file(c, grammar_dir)
        stat_dict = {k: [0] * len(rules) for k in c["bolinas_filters"]}

        first = c.get("first", None)
        last = c.get("last", None)
        for sen_dir in get_range(f"{data_dir}/{c['in_dir']}", first, last):
            for bf in c["bolinas_filters"]:
                fill_stat(bf, c, data_dir, sen_dir, stat_dict)

        stat_dict = OrderedDict(sorted(stat_dict.items()))
        complete_rules, lhs, weights = zip(*rules)

        df_abs = pd.DataFrame(stat_dict, index=range(1, len(rules)+1))
        df_rel = df_abs.apply(lambda x: x*100 / x.sum()).map('{:.4}'.format)
        add_rule_cols(df_abs, complete_rules, lhs, weights)
        df_abs.to_csv(f"{report_dir}/{c['in_dir']}_abs.tsv", sep="\t")
        add_rule_cols(df_rel, complete_rules, lhs, weights)
        df_rel.to_csv(f"{report_dir}/{c['in_dir']}_rel.tsv", sep="\t")

        df_sum_abs = df_abs.iloc[:, 2:].groupby("lhs").sum()
        df_sum_rel = df_sum_abs.apply(lambda x: x*100 / x.sum()).map('{:.4}'.format)
        df_sum_abs.loc["total"] = df_sum_abs.sum()
        df_sum_abs.to_csv(f"{report_dir}/{c['in_dir']}_sum_abs.tsv", sep="\t")
        df_sum_rel.to_csv(f"{report_dir}/{c['in_dir']}_sum_rel.tsv", sep="\t")

        corr_lines = []
        for name, group in df_abs.groupby("lhs"):
            group.to_csv(f"{report_dir}/{c['in_dir']}_abs_{name}.tsv", sep="\t")
            df_lhs = group.iloc[:, -len(stat_dict):].apply(lambda x: x*100 / x.sum()).map('{:.4}'.format)
            add_rule_cols(df_lhs, group["rule"].to_list(), group["lhs"].to_list(), group["weight"].to_list())
            df_lhs.to_csv(f"{report_dir}/{c['in_dir']}_rel_{name}.tsv", sep="\t")

            corr_lines.append(f"{name}\n")
            corr_cols = ["weight"] + list(stat_dict.keys())
            corr_lines.append(f"{corr_cols}\n")
            corr = np.corrcoef(group[corr_cols], rowvar=False)
            corr_lines.append(f"{corr}\n\n")
        with open(f"{report_dir}/{c['in_dir']}_corr.txt", "w") as f:
            f.writelines(corr_lines)


def add_rule_cols(df, complete_rules, lhs, weights):
    df.insert(0, "weight", weights)
    df.insert(1, "rule", complete_rules)
    df.insert(2, "lhs", lhs)


if __name__ == "__main__":
    args = get_data_dir_and_config_args("Script to calculate statistics on rule usage.")
    calc_rule_stat(args.data_dir, args.config)
