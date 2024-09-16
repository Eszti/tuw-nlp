import argparse
import json
from collections import defaultdict

from tuw_nlp.common.eval import f1
from tuw_nlp.sem.hrg.common.report import find_best_in_column, make_markdown_table


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold")
    parser.add_argument("-i", "--in-dir")
    parser.add_argument("-m", "--models")
    parser.add_argument("-rd", "--report-dir")
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


def main(gold_path, in_dir, config_json, report_dir):
    config = json.load(open(config_json))
    gold, gold_multi_rel = get_rels(json.load(open(gold_path)))
    report = "# Pred-Arg Evaluation\n"

    for (table_title, files) in config:
        report += f"## {table_title}\n"
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
    with open(f"{report_dir}/pred_arg_eval.md", "w") as f:
        f.writelines(report)


if __name__ == "__main__":
    args = get_args()
    main(
        args.gold,
        args.in_dir,
        args.models,
        args.report_dir,
    )
