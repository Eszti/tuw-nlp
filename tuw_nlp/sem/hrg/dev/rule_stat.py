import argparse
import json
import os
from collections import OrderedDict

import pandas as pd

from tuw_nlp.sem.hrg.common.io import get_range


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    return parser.parse_args()


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


def main(data_dir, config_json):
    grammar_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/train/grammar"
    report_dir = f"{os.path.dirname(os.path.realpath(__file__))}/reports"

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
        df = pd.DataFrame(stat_dict, index=range(1, len(rules)+1))
        df.insert(0, "lhs", lhs)
        df.insert(1, "rule", complete_rules)
        df.insert(2, "weight", weights)
        df.to_csv(f"{report_dir}/{c['in_dir']}_rule_stat.tsv", sep="\t")


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.config)
