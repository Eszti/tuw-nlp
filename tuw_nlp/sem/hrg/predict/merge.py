import copy
import json
import os
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.io import get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.common.wire_extraction import WiReEx


class Merge(LoopOnSenDirs):

    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json, log=True)
        self.out_dir += f"/{self.config['in_dir']}"
        self.chart_filters = self.config["bolinas_chart_filters"]
        self.postprocess = self.config["postprocess"]
        self.k = self.config.get("k", 0)
        self.all_ex_set = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.ex_stat = defaultdict(lambda: defaultdict(lambda: Counter()))

    def _do_for_sen(self, sen_idx, sen_dir):
        predict_dir = self._get_subdir("predict", create=False, parent_dir=sen_dir)
        for chart_filter in self.chart_filters:
            for pp in self.postprocess:
                subdir_str = f"{chart_filter}_{pp}"
                wire_json = f"{self._add_filter_and_postprocess(predict_dir, chart_filter, pp)}/sen{sen_idx}_wire.json"
                if not os.path.exists(wire_json):
                    for i in range(self.k + 1):
                        self.ex_stat[subdir_str][i][0] += 1
                    continue
                self._get_extractions_for_sen(wire_json, subdir_str)

    def _get_extractions_for_sen(self, wire_json, subdir_str):
        with open(wire_json) as f:
            extractions = json.load(f)
        assert len(extractions.keys()) == 1

        sen = list(extractions.keys())[0]
        all_extractions = extractions[sen]
        all_extractions.sort(key=lambda x: x["k"])

        for i, ex in enumerate(all_extractions):
            wire_ex = WiReEx(ex)
            ex_to_add = copy.copy(wire_ex)
            ex_to_add["extractor"] += "_all"
            self.all_ex_set[subdir_str][0][sen].add(ex_to_add)
            assert i + 1 == wire_ex["k"]
            for j in range(i + 1, self.k + 1):
                ex_to_add = copy.copy(wire_ex)
                ex_to_add["extractor"] += f"_k{j}"
                self.all_ex_set[subdir_str][j][sen].add(ex_to_add)

        for i in range(self.k + 1):
            self.ex_stat[subdir_str][i][len(self.all_ex_set[subdir_str][i][sen])] += 1

    def _after_loop(self):
        for subdir_str, d1 in self.all_ex_set.items():
            chart_filter = subdir_str.split("_")[0]
            pp = subdir_str.split("_")[1]
            for ki, d2 in d1.items():
                if chart_filter in ["prec", "rec", "f1"] and ki > 0:
                    break
                all_ex_list = {}
                for sen, items in d2.items():
                    all_ex_list[sen] = sorted(list(items), key=lambda x: x["k"])

                out_dir = self._add_filter_and_postprocess(self.out_dir, chart_filter, pp)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_fn = self._add_filter_and_postprocess(f"{out_dir}/{self.config['in_dir']}", chart_filter, pp, "_")
                if ki == 0:
                    out_fn += "_all.json"
                else:
                    out_fn += f"_k{ki}.json"

                with open(out_fn, "w") as f:
                    json.dump(all_ex_list, f, indent=4)

                if ki == 0:
                    self._log(f"\nKeeping all extractions.")
                else:
                    self._log(f"\nKeeping top {ki} extractions.")
                sum_sens = 0
                ex_stat_i = {key: v for key, v in sorted(self.ex_stat[subdir_str][ki].items())}
                for j, v in ex_stat_i.items():
                    self._log(f"{j} extraction(s): {v} sen")
                    sum_sens += v
                self._log(f"Sum {sum_sens} sens")
                self._log(f"Output saved to {out_fn}\n")
        super()._after_loop()


if __name__ == "__main__":
    args = get_data_dir_and_config_args("Script to merge predicted wire jsons.")
    Merge(args.data_dir, args.config).run()
