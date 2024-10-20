import argparse
import json
import os

from tuw_nlp.sem.hrg.dev_stat.k_stat import calc_k_stat
from tuw_nlp.sem.hrg.dev_stat.pred_eval import evaluate_predicate_recognition
from tuw_nlp.sem.hrg.dev_stat.rule_stat import calc_rule_stat


def main(data_dir, config_fn):
    config = json.load(open(config_fn))
    config_dir = f"{os.path.dirname(os.path.realpath(__file__))}/config"
    print("Calculate k stat")
    calc_k_stat(data_dir, f"{config_dir}/{config['k_stat_config']}")
    print("Evaluate predicate resolution")
    evaluate_predicate_recognition(data_dir, f"{config_dir}/{config['pred_eval_config']}")
    print("Calculate rule stat")
    calc_rule_stat(data_dir, f"{config_dir}/{config['rule_stat_config']}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.config)
