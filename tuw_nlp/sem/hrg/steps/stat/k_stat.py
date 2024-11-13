import json
import os

import numpy as np
import pandas as pd

from tuw_nlp.sem.hrg.common.report import save_bar_diagram
from tuw_nlp.sem.hrg.common.script.loop_on_models import LoopOnModels


class KStat(LoopOnModels):
    def __init__(self, config=None):
        super().__init__(description="Script to calculate k stat on merged extractions.", config=config)
        self.k_stat_dir = self._get_subdir(f"{self.config_name}", parent_dir=self._get_subdir("stat"))
        self.sen_ids = []
        self.k_values = dict()

    def _before_loop(self):
        self.k_values["gold"] = []
        for sen, extractions in self.gold.items():
            self.sen_ids.append(int(sorted(extractions, key=lambda x: int(x["sen_id"]))[0]["sen_id"]))
            self.k_values["gold"].append(len(extractions))

    def _do_for_model(self, model):
        for chart_filter in sorted(model["bolinas_chart_filters"]):
            for pp in sorted(model["postprocess"]):
                model_name = model["in_dir"]
                all_json = self.__get_all_json(f"{self.in_dir}/{model_name}", chart_filter, pp)
                self.k_values[model_name] = [0] * len(self.sen_ids)
                extractions = json.load(open(all_json))
                for sen, k_extr in extractions.items():
                    assert len(k_extr) > 0
                    sen_id = int(k_extr[-1]["sen_id"])
                    k_extr.sort(key=lambda x: x["k"])
                    k = k_extr[-1]["k"]
                    idx = self.sen_ids.index(sen_id)
                    self.k_values[model_name][idx] = k

    def __get_all_json(self, in_dir, chart_filter, postprocess):
        in_dir = self._add_filter_and_postprocess(in_dir, chart_filter, postprocess)
        files = [i for i in os.listdir(in_dir) if i.endswith("all.json")]
        assert len(files) == 1
        fn = f"{in_dir}/{files[0]}"
        return fn

    def _after_loop(self):
        k_values_df = pd.DataFrame(data=self.k_values, index=self.sen_ids)
        k_values_df.index.name = 'sen_id'
        k_hist = self.__save_bar(k_values_df)
        self.__save_k_values(k_values_df)
        self.__save_k_differences(k_values_df)
        self.__save_hist_corr(k_hist)
        super()._after_loop()

    def __save_bar(self, k_values_df):
        k_hist = k_values_df.apply(lambda x: x.value_counts()).fillna(0).astype(int)
        k_hist.index.name = 'k'
        k_hist.to_csv(f"{self.k_stat_dir}/k_hist.tsv", sep="\t")
        save_bar_diagram(
            k_hist,
            "Number of sentences",
            "Distribution of number of extractions",
            f"{self.k_stat_dir}/k_stat_bar.png"
        )
        return k_hist

    def __save_k_values(self, k_values_df):
        for key in self.k_values:
            if key != "gold":
                k_values_df[f"{key} - gold"] = k_values_df[key] - k_values_df["gold"]
        k_values_df.to_csv(f"{self.k_stat_dir}/k_values.tsv", sep="\t")

    def __save_k_differences(self, k_values_df):
        k_differences = k_values_df.iloc[:, -len(self.models):]\
            .apply(lambda x: x.value_counts(normalize=True)).fillna(0).round(4)
        k_differences.to_csv(f"{self.k_stat_dir}/k_differences.tsv", sep="\t")

    def __save_hist_corr(self, k_hist):
        with open(f"{self.k_stat_dir}/k_corr.txt", "w") as f:
            for model in sorted(self.k_values.keys()):
                if model != "gold":
                    f.writelines(f"{model}\n")
                    corr = np.corrcoef(self.k_values["gold"], self.k_values[model])
                    f.writelines(f"values corr:\n{corr}\n")
                    corr = np.corrcoef(k_hist["gold"], k_hist[model])
                    f.writelines(f"distribution corr:\n{corr}\n")
                    f.writelines(f"sum extractions: {k_hist[model].sum()}\n\n")


if __name__ == "__main__":
    KStat().run()
