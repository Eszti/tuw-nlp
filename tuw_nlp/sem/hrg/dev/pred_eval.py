import argparse
import json
import os
from collections import defaultdict, Counter, OrderedDict

import pandas as pd

from tuw_nlp.common.eval import f1
from tuw_nlp.sem.hrg.common.conll import get_pos_tags
from tuw_nlp.sem.hrg.common.io import get_range, get_k_files_or_assert_all
from tuw_nlp.sem.hrg.common.report import find_best_in_column, make_markdown_table


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold")
    parser.add_argument("-d", "--data-dir")
    parser.add_argument("-c", "--config")
    return parser.parse_args()


def get_rels(extractions):
    ret = defaultdict(set)
    multi_word_rel = 0
    for sen, ex_list in extractions.items():
        for ex in ex_list:
            rel_indexes = ex["rel"]["indexes"]
            if len(rel_indexes) != 1:
                multi_word_rel += 1
            ret[sen].add("_".join([str(idx) for idx in rel_indexes]))
    return ret, multi_word_rel


def calculate_eval_table(data_dir, grammar_dir, chart_filter, pp, gold, gold_multi_rel, report):
    print(f"Processing: {grammar_dir} - {chart_filter} - {pp}")
    if chart_filter or pp:
        report += f"### {chart_filter} - {pp}\n"
    table = [[
        "k",
        "gold rels",
        "avg gold rel / sen",
        "nr gold mult-word rels",
        "predicted rels",
        "avg predicted rel / sen",
        "nr predicted mult-word rels",
        "prec",
        "rec",
        "F1",
    ]]

    in_dir = f"{data_dir}/{grammar_dir}"
    if chart_filter:
        in_dir += f"/{chart_filter}"
    if pp:
        in_dir += f"/{pp}"

    files = sorted([i for i in os.listdir(in_dir) if i.endswith(".json")])
    files = get_k_files_or_assert_all(files)
    for file in files:
        fn = f"{in_dir}/{file}"
        predictions, pred_multi_rel = get_rels(json.load(open(fn)))
        results = {}
        for s, gold_rels in gold.items():
            pred_rels = predictions.get(s, set())
            results[s] = {
                "G & P": len(gold_rels & pred_rels),
                "len G": len(gold_rels),
                "len P": len(pred_rels),
            }
        prec_num, prec_denom = 0, 0
        rec_num, rec_denom = 0, 0
        for s in results.values():
            prec_num += s["G & P"]
            prec_denom += s["len P"]
            rec_num += s["G & P"]
            rec_denom += s["len G"]

        first_col = fn.split(".")[0].split("_")[-1]
        prec = round(prec_num / prec_denom, 4)
        rec = round(rec_num / rec_denom, 4)
        nr_gold_sens = len(gold.keys())
        table.append([
            first_col,
            rec_denom,
            round(rec_denom / nr_gold_sens, 4),
            gold_multi_rel,
            prec_denom,
            round(prec_denom / nr_gold_sens, 4),
            pred_multi_rel,
            prec,
            rec,
            round(f1(prec, rec), 4),
        ])
    bold = find_best_in_column(table, ["prec", "rec", "F1"])
    report += make_markdown_table(table, bold)
    report += "\n"
    return report


def get_all_pos_tags(preproc_dir):
    all_pos_tags = {}
    for sen_dir in get_range(preproc_dir, None, None):
        parsed_doc_file = f"{preproc_dir}/{sen_dir}/preproc/parsed.conll"
        all_pos_tags[sen_dir] = get_pos_tags(parsed_doc_file)
    return all_pos_tags


def get_verb_pos_dist(extractions, pos_tags):
    cnt = Counter()
    for sen, ex_list in extractions.items():
        for ex in ex_list:
            rel_indexes = ex["rel"]["indexes"]
            sen_id = ex["sen_id"]
            for idx in rel_indexes:
                while sen_id not in pos_tags:
                    sen_id -= 1
                cnt[pos_tags[sen_id][str(idx)]] += 1
    table = [[
        "POS tag",
        "number of pred words",
    ]]
    for pos, nr in cnt.items():
        table.append([pos, nr])
    table.append(["sum", sum(cnt.values())])
    return make_markdown_table(table)


def get_all_json(in_dir, chart_filter, pp):
    if chart_filter:
        in_dir += f"/{chart_filter}"
    if pp:
        in_dir += f"/{pp}"
    files = [i for i in os.listdir(in_dir) if i.endswith("all.json")]
    assert len(files) == 1
    fn = f"{in_dir}/{files[0]}"
    return fn


def fill_pos_table(extractions_path, pos_report, pos_tags, model_name):
    model_stat = get_verb_pos_dist(json.load(open(extractions_path)), pos_tags)
    pos_report += f"## {model_name}\n"
    pos_report += f"{model_stat}\n"
    return pos_report


def fill_pred_stat(extractions_path, pred_stat, model_name):
    extractions = json.load(open(extractions_path))
    for sen, ex_list in extractions.items():
        for ex in ex_list:
            sen_id = ex["sen_id"]
            k = ex["k"]
            pred_res = ex["pred_res"]
            pred_stat[f"{sen_id}_k{k}"][model_name] = pred_res


def get_sort_number(x):
    nr1 = int(x.split("_")[0])
    nr2 = int(x.split("_")[1].split("k")[-1])
    return nr1 * 100 + nr2


def save_pred_stat(models, name, pred_stat, report_dir):
    pred_stat_records = []
    models = sorted(list(models))
    pred_stat = OrderedDict(sorted(pred_stat.items(), key=lambda x: get_sort_number(x[0])))
    index = []
    for k, v in pred_stat.items():
        record = []
        for m in models:
            val = v.get(m, "no extraction")
            record.append(val)
        pred_stat_records.append(record)
        index.append(k)
    df = pd.DataFrame(pred_stat_records, columns=models, index=index)
    df = df.fillna("no extraction")
    df.to_csv(f"{report_dir}/{name}_pred_res.tsv", sep="\t")
    df_sum = df.apply(lambda x: x.value_counts()).fillna(0).astype(int)
    df_sum.loc["total"] = df_sum.iloc[:-1].sum()
    df_sum.to_csv(f"{report_dir}/{name}_pred_res_sum.tsv", sep="\t")


def main(gold_path, data_dir, config_json):
    config = json.load(open(config_json))
    report_dir = f"{os.path.dirname(os.path.realpath(__file__))}/reports/pred_eval"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    gold, gold_multi_rel = get_rels(json.load(open(gold_path)))
    eval_report = "# Pred-Arg Evaluation\n"

    pos_tags = get_all_pos_tags(f"{data_dir}/{config['preproc_dir']}")
    pos_report = "# Verb stat\n"
    pos_report = fill_pos_table(gold_path, pos_report, pos_tags, "Dev Gold")

    pred_stat = defaultdict(dict)
    models = set()

    for c in config["models"]:
        if c.get("ignore") and c["ignore"]:
            continue
        eval_report += f"## {c['name']}\n"
        for chart_filter in sorted(c["bolinas_chart_filters"]):
            for pp in sorted(c["postprocess"]):
                eval_report = calculate_eval_table(
                    f"{data_dir}/{config['extractions_dir']}",
                    c["name"],
                    chart_filter,
                    pp,
                    gold,
                    gold_multi_rel,
                    eval_report,
                )

                all_json = get_all_json(f"{data_dir}/{config['extractions_dir']}/{c['name']}", chart_filter, pp)

                pos_report = fill_pos_table(
                    all_json,
                    pos_report,
                    pos_tags,
                    f"{c['name'].split('_')[1]} - {chart_filter}"
                )

                if "random" not in c['name']:
                    model_name = f"{chart_filter}_{pp}"
                    models.add(model_name)
                    fill_pred_stat(all_json, pred_stat, model_name)
        if "random" not in c['name']:
            save_pred_stat(models, c["name"], pred_stat, report_dir)
    with open(f"{report_dir}/pred_retrieval_eval.md", "w") as f:
        f.writelines(eval_report)
    with open(f"{report_dir}/pred_pos_stat.md", "w") as f:
        f.writelines(pos_report)


if __name__ == "__main__":
    args = get_args()
    main(
        args.gold,
        args.data_dir,
        args.config,
    )
