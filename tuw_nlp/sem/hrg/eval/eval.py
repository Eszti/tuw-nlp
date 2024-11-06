import json

from tuw_nlp.sem.hrg.common.io import get_merged_jsons
from tuw_nlp.sem.hrg.common.report import save_pr_curve, find_best_in_column, make_markdown_table
from tuw_nlp.sem.hrg.common.script.loop_on_models import LoopOnModels
from tuw_nlp.sem.hrg.eval.wire_scorer import split_tuples_by_extractor, eval_system, f1


class Eval(LoopOnModels):

    def __init__(self, description):
        super().__init__(description)
        self.test = self.config.get("test", False)
        self.pr_curve = self.config.get("pr_curve", False)
        self.debug = self.config.get("debug", False)
        if self.debug:
            self.temp_dir = self._get_subdir("temp")
        self.eval_md_name = f"{self.config_json.split('/')[-1].split('.json')[0]}"
        self.p_list, self.r_list = [], []
        self.pr_curve_names = []

    def _before_loop(self):
        self.report += "# Evaluation\n"

    def _do_for_model(self, model):
        model_name = model['name']
        self.report += f"## {model_name}\n"
        for chart_filter in sorted(model["bolinas_chart_filters"]):
            for pp in sorted(model["postprocess"]):
                print(f"Processing: {model_name} - {chart_filter} - {pp}")

                mode = "k"
                if self.test:
                    mode = "test"
                elif chart_filter in ["prec", "rec", "f1"]:
                    mode = "all"

                if chart_filter or pp:
                    self.report += f"### {chart_filter} - {pp}\n"

                in_dir = f"{self.in_dir}/{model['in_dir']}"
                files = get_merged_jsons(
                    in_dir,
                    chart_filter,
                    pp,
                    only_all=(mode == "all")
                )

                p, r = [], []
                self.__calculate_table(files, p, r, mode)

                if len(files) > 1:
                    self.p_list.append(p)
                    self.r_list.append(r)
                    self.pr_curve_names.append(f"{model_name}-{chart_filter}-{pp}")

    def __calculate_table(self, files, p, r, mode):
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

            predictions_by_model = split_tuples_by_extractor(self.gold.keys(), all_predictions)
            for model, system_extractions in sorted(predictions_by_model.items()):
                metrics, raw_match_scores, exact_matches, matches = eval_system(self.gold, system_extractions)

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

                if self.debug:
                    prec_l = [m[2]["prec"] for m in matches]
                    rec_l = [m[2]["rec"] for m in matches]
                    print(f"model: {model}")
                    print(f"avg prec: {sum(prec_l) / len(prec_l)}")
                    print(f"avg rec: {sum(rec_l) / len(rec_l)}\n")
                    with open(f"{self.temp_dir}/matches_{file}", "w") as f:
                        json.dump(matches, f, indent=4)
                    with open(f"{self.temp_dir}/exact_matches_{file}", "w") as f:
                        json.dump(exact_matches, f, indent=4)
                    with open(f"{self.temp_dir}/{model}_prec_scores.dat", "w") as f:
                        f.write(str(raw_match_scores[0]))
                    with open(f"{self.temp_dir}/{model}_rec_scores.dat", "w") as f:
                        f.write(str(raw_match_scores[1]))

        bold = find_best_in_column(table, ["prec", "rec", "F1"])
        self.report += make_markdown_table(table, bold)
        self.report += "\n"

    def _after_loop(self):
        if self.pr_curve and not self.test:
            save_pr_curve(
                self.p_list,
                self.r_list,
                self.pr_curve_names,
                f"{self.report_dir}/pr_curve_{self.eval_md_name}.png"
            )
            self.report += f"## P-R curve\n![](pr_curve_{self.eval_md_name}.png)"

        with open(f"{self.report_dir}/{self.eval_md_name}.md", "w") as f:
            f.writelines(self.report)
        super()._after_loop()


if __name__ == "__main__":
    Eval("Script to evaluate systems.").run()
