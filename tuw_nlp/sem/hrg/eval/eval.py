import json
import os

from tuw_nlp.sem.hrg.common.io import get_merged_jsons, get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.report import save_pr_curve, find_best_in_column, make_markdown_table
from tuw_nlp.sem.hrg.eval.wire_scorer import split_tuples_by_extractor, eval_system, f1


def calculate_table(files, gold, report, p, r, mode, debug, temp_dir):
    first_col = "k" if mode == "k" else "model"
    table = [[first_col,
              "predicted extractions",
              "gold extractions",
              "matches",
              "exact matches",
              "prec",
              "rec",
              "F1"]]

    for file in sorted(files):
        all_predictions = json.load(open(file))

        predictions_by_model = split_tuples_by_extractor(gold.keys(), all_predictions)
        for model, system_extractions in sorted(predictions_by_model.items()):
            metrics, raw_match_scores, exact_matches, matches = eval_system(gold, system_extractions)

            prec, rec = metrics['precision'], metrics['recall']
            f1_score = round(f1(prec, rec), 4)
            prec, rec = round(prec, 4), round(rec, 4)
            p.append(prec)
            r.append(rec)

            first_col = model
            if mode == "k":
                first_col = model.split("_")[-1]
            elif mode == "all":
                first_col = "all"
            pred_extractions = metrics['exactmatches_precision'][1]
            nr_matches = metrics['matches']
            nr_exact_matches = metrics['exactmatches_precision'][0]
            gold_extractions = metrics['exactmatches_recall'][1]

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

            if debug:
                prec_l = [m[2]["prec"] for m in matches]
                rec_l = [m[2]["rec"] for m in matches]
                print(f"model: {model}")
                print(f"avg prec: {sum(prec_l) / len(prec_l)}")
                print(f"avg rec: {sum(rec_l) / len(rec_l)}\n")
                with open(f"{temp_dir}/matches_{file}", "w") as f:
                    json.dump(matches, f, indent=4)
                with open(f"{temp_dir}/exact_matches_{file}", "w") as f:
                    json.dump(exact_matches, f, indent=4)
                with open(f"{temp_dir}/{model}_prec_scores.dat", "w") as f:
                    f.write(str(raw_match_scores[0]))
                with open(f"{temp_dir}/{model}_rec_scores.dat", "w") as f:
                    f.write(str(raw_match_scores[1]))

    bold = find_best_in_column(table, ["prec", "rec", "F1"])
    report += make_markdown_table(table, bold)
    report += "\n"
    return report


def main(data_dir, config_json):
    config = json.load(open(config_json))
    report_dir = f"{os.path.dirname(os.path.realpath(__file__))}/reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    temp_dir = f"{os.path.dirname(os.path.realpath(__file__))}/temp"

    gold_path = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data/{config['gold_fn']}"
    gold = json.load(open(gold_path))

    test = config.get("test", False)
    pr_curve = config.get("pr_curve", False)
    debug = config.get("debug", False)

    if debug:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    report = "# Evaluation\n"

    p_list, r_list = [], []
    pr_curve_names = []
    for grammar_name, c in config["models"].items():
        if c.get("ignore") and c["ignore"]:
            continue
        report += f"## {grammar_name}\n"
        for chart_filter in sorted(c["bolinas_chart_filters"]):
            for pp in sorted(c["postprocess"]):
                print(f"Processing: {grammar_name} - {chart_filter} - {pp}")

                mode = "k"
                if test:
                    mode = "test"
                elif chart_filter in ["prec", "rec", "f1"]:
                    mode = "all"

                if chart_filter or pp:
                    report += f"### {chart_filter} - {pp}\n"

                in_dir = f"{data_dir}/{config['extractions_dir']}/{c['in_dir']}"
                files = get_merged_jsons(
                    in_dir,
                    chart_filter,
                    pp,
                    only_all=(mode == "all")
                )

                p, r = [], []
                report = calculate_table(files, gold, report, p, r, mode, debug, temp_dir)

                if len(files) > 1:
                    p_list.append(p)
                    r_list.append(r)
                    pr_curve_names.append(f"{grammar_name}-{chart_filter}-{pp}")

    eval_md_name = f"{config_json.split('/')[-1].split('.json')[0]}"

    if pr_curve and not test:
        save_pr_curve(p_list, r_list, pr_curve_names, f"{report_dir}/pr_curve_{eval_md_name}.png")
        report += f"## P-R curve\n![](pr_curve_{eval_md_name}.png)"

    with open(f"{report_dir}/{eval_md_name}.md", "w") as f:
        f.writelines(report)


if __name__ == "__main__":
    args = get_data_dir_and_config_args("Script to evaluate systems.")
    main(args.data_dir, args.config)
