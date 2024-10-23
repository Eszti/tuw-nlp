import copy
import json
import os
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.io import get_range, get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.sem.hrg.common.wire_extraction import WiReEx


class MergeScript(Script):

    def run_loop(self):
        for chart_filter in self.config["bolinas_chart_filters"]:
            for pp in self.config["postprocess"]:
                self._merge(chart_filter, pp)

    def _merge(self, chart_filter, postprocess):
        k = self.config.get("k", 0)
        all_ex_set = defaultdict(lambda: defaultdict(set))
        ex_stat = defaultdict(lambda: Counter())
        in_path = f"{self.data_dir}/{self.config['in_dir']}"
        for sen_dir in get_range(in_path, self.first, self.last):
            sen_dir = str(sen_dir)
            predict_dir = os.path.join(in_path, sen_dir, "predict")
            if chart_filter:
                predict_dir += f"/{chart_filter}"
            if postprocess:
                predict_dir += f"/{postprocess}"
            wire_json = f"{predict_dir}/sen{sen_dir}_wire.json"

            if not os.path.exists(wire_json):
                for i in range(k + 1):
                    ex_stat[i][0] += 1
                continue

            self._get_extractions_for_sen(all_ex_set, ex_stat, k, wire_json)
        for ki, d in all_ex_set.items():
            if chart_filter in ["prec", "rec", "f1"] and ki > 0:
                break
            all_ex_list = {}
            for sen, items in d.items():
                all_ex_list[sen] = sorted(list(items), key=lambda x: x["k"])

            out_fn_dir = f"{self.data_dir}/{self.config['out_dir']}/{self.config['in_dir']}"
            if chart_filter:
                out_fn_dir += f"/{chart_filter}"
            if postprocess:
                out_fn_dir += f"/{postprocess}"
            if not os.path.exists(out_fn_dir):
                os.makedirs(out_fn_dir)
            out_fn_k = f"{out_fn_dir}/{self.config['in_dir']}"
            if chart_filter:
                out_fn_k += f"_{chart_filter}"
            if postprocess:
                out_fn_k += f"_{postprocess}"
            if ki == 0:
                out_fn_k += "_all.json"
            else:
                out_fn_k += f"_k{ki}.json"

            with open(out_fn_k, "w") as f:
                json.dump(all_ex_list, f, indent=4)

            if ki == 0:
                print(f"\nKeeping all extractions.")
            else:
                print(f"\nKeeping top {ki} extractions.")
            sum = 0
            ex_stat_i = {key: v for key, v in sorted(ex_stat[ki].items())}
            for j, v in ex_stat_i.items():
                print(f"{j} extraction(s): {v} sen")
                sum += v
            print(f"Sum {sum} sens")
            print(f"Output saved to {out_fn_k}\n")

    @staticmethod
    def _get_extractions_for_sen(all_ex_set, ex_stat, k, wire_json):
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
            all_ex_set[0][sen].add(ex_to_add)
            assert i + 1 == wire_ex["k"]
            for j in range(i + 1, k + 1):
                ex_to_add = copy.copy(wire_ex)
                ex_to_add["extractor"] += f"_k{j}"
                all_ex_set[j][sen].add(ex_to_add)
        for i in range(k + 1):
            ex_stat[i][len(all_ex_set[i][sen])] += 1


if __name__ == "__main__":
    args = get_data_dir_and_config_args("Script to merge predicted wire jsons.")
    script = MergeScript(
        args.data_dir,
        args.config,
    )
    script.run()
