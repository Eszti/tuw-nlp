import json
from collections import defaultdict, Counter, OrderedDict

import pandas as pd

from tuw_nlp.common.eval import f1
from tuw_nlp.sem.hrg.common.conll import ConllSen
from tuw_nlp.sem.hrg.common.report import find_best_in_column, make_markdown_table
from tuw_nlp.sem.hrg.common.script.loop_on_models import LoopOnModels


class PredEval(LoopOnModels):
    def __init__(self, config=None):
        super().__init__(description="Script to evaluate predicate resolution.", config=config)
        self.pred_eval_dir = self._get_subdir(f"{self.config_name}", self._get_subdir("stat"))
        self.preproc_dir = f"{self.data_dir}/{self.config['preproc_dir']}"

    def _before_loop(self):
        self.gold_pred_indices, self.gold_multi_rel = self.__get_rels(self.gold)
        self.eval_report = "# Pred-Arg Evaluation\n"

        self.pos_tags = self.__get_all_pos_tags()
        self.pos_report = "# Verb stat\n"
        self.__fill_pos_table(self.gold, "Dev Gold")

    def _do_for_model(self, model):
        model_name = model['name']
        processed_models = set()
        pred_stat = defaultdict(dict)
        self.eval_report += f"## {model_name}\n"
        for chart_filter in sorted(model["bolinas_chart_filters"]):
            for pp in sorted(model["postprocess"]):
                self.__calculate_eval_table(model["name"], chart_filter, pp)
                all_json = self._get_merged_jsons(
                    f"{self.in_dir}/{model_name}",
                    chart_filter,
                    pp,
                    only_all=True
                )[0]
                extractions = json.load(open(all_json))
                self.__fill_pos_table(extractions, f"{model_name.split('_')[1]} - {chart_filter}")
                if "random" not in model_name:
                    processed_model_name = f"{chart_filter}_{pp}"
                    processed_models.add(processed_model_name)
                    self.__fill_pred_stat(extractions, pred_stat, processed_model_name)
        if "random" not in model_name:
            self.__save_pred_stat(model_name, pred_stat, processed_models)

    def __get_all_pos_tags(self):
        all_pos_tags = {}
        for sen_dir in self._get_all_sen_dirs(self.preproc_dir):
            sen_idx = int(sen_dir.split("/")[-1])
            all_pos_tags[sen_idx] = ConllSen(sen_dir).pos_tags()
        return all_pos_tags

    def __calculate_eval_table(self, model_name, chart_filter, pp):
        print(f"Processing: {model_name} - {chart_filter} - {pp}")
        if chart_filter or pp:
            self.eval_report += f"### {chart_filter} - {pp}\n"
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

        in_dir = f"{self.in_dir}/{model_name}"
        files = self._get_merged_jsons(in_dir, chart_filter, pp)
        for file in files:
            predictions, pred_multi_rel = self.__get_rels(json.load(open(file)))
            results = {}
            for s, gold_rels in self.gold_pred_indices.items():
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

            first_col = file.split("/")[-1].split(".")[0].split("_")[-1]
            prec = round(prec_num / prec_denom, 4)
            rec = round(rec_num / rec_denom, 4)
            nr_gold_sens = len(self.gold_pred_indices.keys())
            nr_pred_sens = len(predictions.keys())
            table.append([
                first_col,
                rec_denom,
                round(rec_denom / nr_gold_sens, 4),
                self.gold_multi_rel,
                prec_denom,
                round(prec_denom / nr_pred_sens, 4),
                pred_multi_rel,
                prec,
                rec,
                round(f1(prec, rec), 4),
            ])
        bold = find_best_in_column(table, ["prec", "rec", "F1"])
        self.eval_report += make_markdown_table(table, bold)
        self.eval_report += "\n"

    @staticmethod
    def __get_rels(extractions):
        ret = defaultdict(set)
        multi_word_rel = 0
        for sen, ex_list in extractions.items():
            for ex in ex_list:
                rel_indexes = ex["rel"]["indexes"]
                if len(rel_indexes) != 1:
                    multi_word_rel += 1
                ret[sen].add("_".join([str(idx) for idx in sorted(rel_indexes)]))
        return ret, multi_word_rel

    def __fill_pos_table(self, extractions, model_name):
        self.pos_report += f"## {model_name}\n"
        cnt = Counter()
        for sen, ex_list in extractions.items():
            for ex in ex_list:
                rel_indexes = ex["rel"]["indexes"]
                sen_id = ex["sen_id"]
                if self.last and sen_id > self.last:
                    break
                for idx in rel_indexes:
                    while sen_id not in self.pos_tags:
                        sen_id -= 1
                    cnt[self.pos_tags[sen_id][idx-1]] += 1
        table = [[
            "POS tag",
            "number of pred words",
        ]]
        for pos, nr in cnt.items():
            table.append([pos, nr])
        table.append(["sum", sum(cnt.values())])
        self.pos_report += f"{make_markdown_table(table)}\n"

    @staticmethod
    def __fill_pred_stat(extractions, pred_stat, model_name):
        for sen, ex_list in extractions.items():
            for ex in ex_list:
                sen_id = ex["sen_id"]
                k = ex["k"]
                pred_res = ex["pred_res"]
                pred_stat[f"{sen_id}_k{k}"][model_name] = pred_res

    def __save_pred_stat(self, name, pred_stat, models):
        pred_stat_records = []
        models = sorted(list(models))
        pred_stat = OrderedDict(sorted(pred_stat.items(), key=lambda x: self.__get_sort_number(x[0])))
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
        df.to_csv(f"{self.pred_eval_dir}/{name}_pred_res.tsv", sep="\t")
        df_sum = df.apply(lambda x: x.value_counts()).fillna(0).astype(int)
        df_sum.loc["total"] = df_sum.sum()
        if "no_extraction" in df_sum:
            df_sum.loc["total"] = df_sum["total"] - df_sum["no_extraction"]
        df_sum.to_csv(f"{self.pred_eval_dir}/{name}_pred_res_sum.tsv", sep="\t")

    @staticmethod
    def __get_sort_number(x):
        nr1 = int(x.split("_")[0])
        nr2 = int(x.split("_")[1].split("k")[-1])
        return nr1 * 100 + nr2

    def _after_loop(self):
        with open(f"{self.pred_eval_dir}/pred_retrieval_eval.md", "w") as f:
            f.writelines(self.eval_report)
        with open(f"{self.pred_eval_dir}/pred_pos_stat.md", "w") as f:
            f.writelines(self.pos_report)
        super()._after_loop()


if __name__ == "__main__":
    PredEval().run()
