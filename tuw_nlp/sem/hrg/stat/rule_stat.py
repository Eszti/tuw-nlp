import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from tuw_nlp.sem.hrg.common.script.loop_on_models import LoopOnModels


class RuleStat(LoopOnModels):
    def __init__(self, config=None):
        super().__init__(description="Script to calculate statistics on rule usage.", config=config)
        self.rule_stat_dir = self._get_subdir(f"{self.config_name}", self.report_dir)
        self.grammar_dir = f"{os.path.dirname(self.parent_dir)}/train/grammar"

    def _do_for_model(self, model):
        rules = self.__get_rules_from_grammar_file(f"{self.grammar_dir}/{model['grammar_file']}")
        stat_dict = {bolinas_filter: [0] * len(rules) for bolinas_filter in model["bolinas_filters"]}

        for sen_dir in self._get_all_sen_dirs(f"{self.data_dir}/{model['in_dir']}"):
            for bolinas_filter in model["bolinas_filters"]:
                sen_idx = int(sen_dir.split("/")[-1])
                self.__fill_stat(
                    f"{sen_dir}/bolinas/{bolinas_filter}/sen{sen_idx}_derivation.txt",
                    stat_dict[bolinas_filter]
                )

        stat_dict = OrderedDict(sorted(stat_dict.items()))
        complete_rules, lhs, weights = zip(*rules)

        df_abs = pd.DataFrame(stat_dict, index=range(1, len(rules) + 1))
        df_rel = df_abs.apply(lambda x: x * 100 / x.sum()).map('{:.4}'.format)
        self.__add_rule_cols(df_abs, complete_rules, lhs, weights)
        df_abs.to_csv(f"{self.rule_stat_dir}/{model['in_dir']}_abs.tsv", sep="\t")
        self.__add_rule_cols(df_rel, complete_rules, lhs, weights)
        df_rel.to_csv(f"{self.rule_stat_dir}/{model['in_dir']}_rel.tsv", sep="\t")

        df_sum_abs = df_abs.iloc[:, 2:].groupby("lhs").sum()
        df_sum_rel = df_sum_abs.apply(lambda x: x * 100 / x.sum()).map('{:.4}'.format)
        df_sum_abs.loc["total"] = df_sum_abs.sum()
        df_sum_abs.to_csv(f"{self.rule_stat_dir}/{model['in_dir']}_sum_abs.tsv", sep="\t")
        df_sum_rel.to_csv(f"{self.rule_stat_dir}/{model['in_dir']}_sum_rel.tsv", sep="\t")

        corr_lines = []
        for name, group in df_abs.groupby("lhs"):
            group.to_csv(f"{self.rule_stat_dir}/{model['in_dir']}_abs_{name}.tsv", sep="\t")
            df_lhs = group.iloc[:, -len(stat_dict):].apply(lambda x: x * 100 / x.sum()).map('{:.4}'.format)
            self.__add_rule_cols(df_lhs, group["rule"].to_list(), group["lhs"].to_list(), group["weight"].to_list())
            df_lhs.to_csv(f"{self.rule_stat_dir}/{model['in_dir']}_rel_{name}.tsv", sep="\t")

            corr_lines.append(f"{name}\n")
            corr_cols = ["weight"] + list(stat_dict.keys())
            corr_lines.append(f"{corr_cols}\n")
            corr = np.corrcoef(group[corr_cols], rowvar=False)
            corr_lines.append(f"{corr}\n\n")
        with open(f"{self.rule_stat_dir}/{model['in_dir']}_corr.txt", "w") as f:
            f.writelines(corr_lines)

    @staticmethod
    def __get_rules_from_grammar_file(grammar_fn):
        with open(grammar_fn) as f:
            lines = f.readlines()
            rules = [(
                l.split("\t")[0].split(";")[0],
                l.split("\t")[0].split(";")[0].split(" ")[0],
                float(l.split("\t")[1]),
            ) for l in lines]
        return rules

    @staticmethod
    def __fill_stat(derivation_file, stat_dict):
        if os.path.exists(derivation_file):
            with open(derivation_file) as f:
                lines = f.readlines()
                for line in lines:
                    if "#" in line or not line or line == "\n":
                        continue
                    rule_id = int(line.split("\t")[0])
                    stat_dict[rule_id - 1] += 1

    @staticmethod
    def __add_rule_cols(df, complete_rules, lhs, weights):
        df.insert(0, "weight", weights)
        df.insert(1, "rule", complete_rules)
        df.insert(2, "lhs", lhs)


if __name__ == "__main__":
    RuleStat().run()
