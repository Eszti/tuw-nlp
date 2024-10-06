import argparse
import json
import os

from tuw_nlp.sem.hrg.common.report import save_pr_curve, find_best_in_column, make_markdown_table
from tuw_nlp.sem.hrg.eval.wire_scorer import check_keys, keep_only_common, split_tuples_by_extractor, eval_system, f1


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold")
    parser.add_argument("-d", "--data-dir")
    parser.add_argument("-c", "--config")
    parser.add_argument("-rd", "--report-dir")
    parser.add_argument("-td", "--temp-dir")
    parser.add_argument("-oc", "--only-common", action="store_true")
    parser.add_argument("-r", "--raw-scores", action="store_true")
    return parser.parse_args()


def calculate_table(
        data_dir,
        grammar_dir,
        chart_filter,
        pp,
        gold,
        only_common,
        p_list,
        r_list,
        pr_curve_names,
        grammar_name,
        raw_scores,
        report,
        temp_dir,
        test,
):
    print(f"Processing: {grammar_dir} - {chart_filter} - {pp}")
    if chart_filter or pp:
        report += f"### {chart_filter} - {pp}\n"
    first_col = "model" if test else "k"
    table = [[first_col,
              "predicted extractions",
              "gold extractions",
              "matches",
              "exact matches",
              "prec",
              "rec",
              "F1"]]
    p, r = [], []
    in_dir = f"{data_dir}/{grammar_dir}"
    if chart_filter:
        in_dir += f"/{chart_filter}"
    if pp:
        in_dir += f"/{pp}"
    files = sorted([i for i in os.listdir(in_dir) if i.endswith(".json")])
    if not test:
        k_files = [i for i in files if i.split("_")[-1].startswith("k")]
        if k_files:
            files = sorted(k_files, key=lambda x: int(x.split('.')[0].split("_")[-1].split("k")[-1]))
        else:
            assert len(files) == 1 and files[0].endswith("_all.json")
    for file in files:
        fn = f"{in_dir}/{file}"
        print(fn)
        all_predictions = json.load(open(fn))

        if only_common:
            common = check_keys(gold.keys(), all_predictions.keys())
            keep_only_common(gold, common)
            keep_only_common(all_predictions, common)

        predictions_by_model = split_tuples_by_extractor(gold.keys(), all_predictions)
        for model, system_extractions in predictions_by_model.items():
            metrics, raw_match_scores, exact_matches, matches = eval_system(gold, system_extractions)

            if raw_scores:
                with open(f"{temp_dir}/{model}_prec_scores.dat", "w") as f:
                    f.write(str(raw_match_scores[0]))
                with open(f"{temp_dir}/{model}_rec_scores.dat", "w") as f:
                    f.write(str(raw_match_scores[1]))

            prec, rec = metrics['precision'], metrics['recall']
            f1_score = round(f1(prec, rec), 4)
            prec, rec = round(prec, 4), round(rec, 4)
            p.append(prec)
            r.append(rec)
            pred_extractions = metrics['exactmatches_precision'][1]
            nr_matches = metrics['matches']
            nr_exact_matches = metrics['exactmatches_precision'][0]
            gold_extractions = metrics['exactmatches_recall'][1]

            first_col = model if test else model.split("_")[-1]
            table.append([first_col,
                          pred_extractions,
                          gold_extractions,
                          nr_matches,
                          nr_exact_matches,
                          prec,
                          rec,
                          f1_score])
            assert nr_exact_matches == len(exact_matches)
            assert nr_matches == len(matches)
            prec_l = [m[2]["prec"] for m in matches]
            rec_l = [m[2]["rec"] for m in matches]
            print(f"model: {model}")
            print(f"avg prec: {sum(prec_l) / len(prec_l)}")
            print(f"avg rec: {sum(rec_l) / len(rec_l)}\n")
            with open(f"{temp_dir}/matches_{file}", "w") as f:
                json.dump(matches, f, indent=4)
            with open(f"{temp_dir}/exact_matches_{file}", "w") as f:
                json.dump(exact_matches, f, indent=4)
    bold = find_best_in_column(table, ["prec", "rec", "F1"])
    report += make_markdown_table(table, bold)
    report += "\n"
    if len(files) > 1:
        p_list.append(p)
        r_list.append(r)
        pr_curve_names.append(f"{grammar_name}-{chart_filter}-{pp}")
    return report


def main(gold_path, data_dir, config_json, only_common, raw_scores, report_dir, temp_dir):
    config = json.load(open(config_json))
    gold = json.load(open(gold_path))
    report = "# Evaluation\n"

    p_list, r_list = [], []
    pr_curve_names = []
    for grammar_name, c in config["models"].items():
        if c.get("ignore") and c["ignore"]:
            continue
        report += f"## {grammar_name}\n"
        for chart_filter in sorted(c["bolinas_chart_filters"]):
            for pp in sorted(c["postprocess"]):
                report = calculate_table(
                    data_dir,
                    c["in_dir"],
                    chart_filter,
                    pp,
                    gold,
                    only_common,
                    p_list,
                    r_list,
                    pr_curve_names,
                    grammar_name,
                    raw_scores,
                    report,
                    temp_dir,
                    config["test"]
                )
    eval_md_name = config_json.split("/")[-1].split("_config.json")[0]
    if config["pr_curve"] and not config["test"]:
        save_pr_curve(p_list, r_list, pr_curve_names, f"{report_dir}/pr_curve_{eval_md_name}.png")
        report += f"## P-R curve\n![](pr_curve_{eval_md_name}.png)"
    with open(f"{report_dir}/{eval_md_name}.md", "w") as f:
        f.writelines(report)


if __name__ == "__main__":
    args = get_args()
    main(
        args.gold,
        args.data_dir,
        args.config,
        args.only_common,
        args.raw_scores,
        args.report_dir,
        args.temp_dir
    )