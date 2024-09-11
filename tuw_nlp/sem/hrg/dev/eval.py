import argparse
import json

from tuw_nlp.sem.hrg.common.report import save_pr_curve, find_best_in_column, make_markdown_table
from tuw_nlp.sem.hrg.eval.wire_scorer import check_keys, keep_only_common, split_tuples_by_extractor, eval_system, f1

config = {
    "models": [
        (
            "Grammar 99",
            [
                "dev_gr99_k1.json",
                "dev_gr99_k2.json",
                "dev_gr99_k3.json",
                "dev_gr99_k4.json",
                "dev_gr99_k5.json",
                "dev_gr99_k6.json",
                "dev_gr99_k7.json",
                "dev_gr99_k8.json",
                "dev_gr99_k9.json",
                "dev_gr99_k10.json",
            ]
        ),
        (
            "Grammar 100",
            [
                "dev_gr100_k1.json",
                "dev_gr100_k2.json",
                "dev_gr100_k3.json",
                "dev_gr100_k4.json",
                "dev_gr100_k5.json",
                "dev_gr100_k6.json",
                "dev_gr100_k7.json",
                "dev_gr100_k8.json",
                "dev_gr100_k9.json",
                "dev_gr100_k10.json",
            ]
        ),
        (
            "Grammar 200",
            [
                "dev_gr200_k1.json",
                "dev_gr200_k2.json",
                "dev_gr200_k3.json",
                "dev_gr200_k4.json",
                "dev_gr200_k5.json",
                "dev_gr200_k6.json",
                "dev_gr200_k7.json",
                "dev_gr200_k8.json",
                "dev_gr200_k9.json",
                "dev_gr200_k10.json",
            ]
        ),
        (
            "Random ",
            [
                "dev_grRND_all.json",
            ]
        ),
    ]
}


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold")
    parser.add_argument("-i", "--in-dir")
    parser.add_argument("-rd", "--report-dir")
    parser.add_argument("-td", "--temp-dir")
    parser.add_argument("-c", "--only-common", action="store_true")
    parser.add_argument("-r", "--raw-scores", action="store_true")
    return parser.parse_args()


def main(gold_path, in_dir, only_common, raw_scores, report_dir, temp_dir):
    gold = json.load(open(gold_path))
    report = "# Evaluation\n"

    p_list, r_list = [], []
    for (model, files) in config["models"]:
        report += f"## {model}\n"
        table = [["variation", "predicted extractions", "gold extractions", "matches", "exact matches", "prec", "rec", "F1"]]
        p, r = [], []

        for file in files:
            fn = f"{in_dir}/{file}"
            variation = fn.split(".")[0].split("_")[-1]
            all_predictions = json.load(open(fn))

            if only_common:
                common = check_keys(gold.keys(), all_predictions.keys())
                keep_only_common(gold, common)
                keep_only_common(all_predictions, common)

            predictions_by_model = split_tuples_by_extractor(gold.keys(), all_predictions)
            models = predictions_by_model.keys()
            assert len(models) == 1

            m = next(iter(models))
            metrics, raw_match_scores, exact_matches, matches = eval_system(gold, predictions_by_model[m])

            if raw_scores:
                with open("raw_scores/" + m + "_prec_scores.dat", "w") as f:
                    f.write(str(raw_match_scores[0]))
                with open("raw_scores/" + m + "_rec_scores.dat", "w") as f:
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
            table.append([variation, pred_extractions, gold_extractions, nr_matches, nr_exact_matches, prec, rec, f1_score])
            # assert nr_exact_matches == len(exact_matches)
            assert nr_matches == len(matches)
            prec_l = [m[2]["prec"] for m in matches]
            rec_l = [m[2]["rec"] for m in matches]
            print(f"model: {model}, variation: {variation}")
            print(f"avg prec: {sum(prec_l)/len(prec_l)}")
            print(f"avg rec: {sum(rec_l)/len(rec_l)}\n")
            with open(f"{temp_dir}/matches_{file}", "w") as f:
                json.dump(matches, f, indent=4)
            with open(f"{temp_dir}/exact_matches_{file}", "w") as f:
                json.dump(exact_matches, f, indent=4)
        bold = find_best_in_column(table, ["prec", "rec", "F1"])
        report += make_markdown_table(table, bold)
        report += "\n"
        p_list.append(p)
        r_list.append(r)
    save_pr_curve(p_list, r_list, [model[0] for model in config["models"]], f"{report_dir}/pr_curve.png")
    report += f"## P-R curve\n![](pr_curve.png)"
    with open(f"{report_dir}/eval.md", "w") as f:
        f.writelines(report)


if __name__ == "__main__":
    args = get_args()
    main(args.gold,
         args.in_dir,
         args.only_common,
         args.raw_scores,
         args.report_dir,
         args.temp_dir)
