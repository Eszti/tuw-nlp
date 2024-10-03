import argparse
import json
import os
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.wire_extraction import WiReEx


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    return parser.parse_args()


def get_extractions_for_sen(all_ex_set, ex_stat, k, wire_json):
    with open(wire_json) as f:
        extractions = json.load(f)
    assert len(extractions.keys()) == 1
    sen = list(extractions.keys())[0]
    all_extractions = extractions[sen]
    all_extractions.sort(key=lambda x: x["k"])
    for i, ex in enumerate(all_extractions):
        wire_ex = WiReEx(ex)
        all_ex_set[0][sen].add(wire_ex)
        assert i + 1 == wire_ex["k"]
        for j in range(i + 1, k + 1):
            all_ex_set[j][sen].add(wire_ex)
    for i in range(k+1):
        ex_stat[i][len(all_ex_set[i][sen])] += 1


def merge(data_dir, in_dir, out_dir, chart_filter, postprocess, k, first, last):
    all_ex_set = defaultdict(lambda: defaultdict(set))
    ex_stat = defaultdict(lambda: Counter())
    in_path = f"{data_dir}/{in_dir}"
    for sen_dir in get_range(in_path, first, last):
        sen_dir = str(sen_dir)
        predict_dir = os.path.join(in_path, sen_dir, "predict")
        if chart_filter:
            predict_dir += f"/{chart_filter}"
        if postprocess:
            predict_dir += f"/{postprocess}"
        wire_json = f"{predict_dir}/sen{sen_dir}_wire.json"

        if not os.path.exists(wire_json):
            for i in range(k+1):
                ex_stat[i][0] += 1
            continue

        get_extractions_for_sen(all_ex_set, ex_stat, k, wire_json)
    for ki, d in all_ex_set.items():
        all_ex_list = {}
        for sen, items in d.items():
            for item in items:
                item["extractor"] = item["extractor"].split("_")[0] + f"_k{ki}"
            all_ex_list[sen] = sorted(list(items), key=lambda x: x["k"])

        out_fn_dir = f"{data_dir}/{out_dir}/{in_dir}"
        if chart_filter:
            out_fn_dir += f"/{chart_filter}"
        if postprocess:
            out_fn_dir += f"/{postprocess}"
        if not os.path.exists(out_fn_dir):
            os.makedirs(out_fn_dir)
        out_fn_k = f"{out_fn_dir}/{in_dir}"
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


def main(data_dir, config_json):
    config = json.load(open(config_json))
    for in_dir, c in config.items():
        if c.get("ignore") and c["ignore"]:
            continue
        first = c.get("first", None)
        last = c.get("last", None)
        k = c.get("k", 0)
        for chart_filter in c["bolinas_chart_filters"]:
            for pp in c["postprocess"]:
                merge(data_dir, in_dir, c["out_dir"], chart_filter, pp, k, first, last)


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.config)
