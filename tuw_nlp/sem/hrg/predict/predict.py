import json
import logging
import os
from collections import defaultdict
from tuw_nlp.sem.hrg.common.conll import get_pos_tags, get_sen_txt
from tuw_nlp.sem.hrg.common.io import get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction
from tuw_nlp.sem.hrg.postproc.postproc import postprocess


class Predict(LoopOnSenDirs):

    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json, log=True)
        self.out_dir += self.in_dir
        self.preproc_dir = f"{self.data_dir}/{self.config['preproc_dir']}"
        self.chart_filters = self.config["bolinas_chart_filters"]
        self.postprocess = self.config["postprocess"]
        self.k = self.config.get("k", 0)

    def _do_for_sen(self, sen_idx, sen_dir):
        sen_text, pos_tags, top_order, orig_conll = self.__get_preproc_input(sen_idx)
        predict_dir = self._get_subdir("predict", sen_dir)
        for chart_filter in self.chart_filters:
            chart_filter_dir = f"{predict_dir}/{chart_filter}"
            if not os.path.exists(chart_filter_dir):
                os.makedirs(chart_filter_dir)

            labels_file = f"{sen_dir}/bolinas/{chart_filter}/sen{sen_idx}_predicted_labels.txt"
            bolinas_extractions = []
            if os.path.exists(labels_file):
                with open(labels_file) as f:
                    bolinas_extractions = f.readlines()

            wire_extractions = defaultdict(lambda: defaultdict(list))
            for i, labels_line in enumerate(bolinas_extractions):
                k = i + 1
                labels_str = labels_line.split(';')[0].strip()
                score = labels_line.split(';')[1].strip()

                for pp in self.postprocess:
                    extracted_labels = json.loads(labels_str)
                    out_dir = self._add_filter_and_postprocess(predict_dir, chart_filter, pp)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    pred_res, extracted_labels_all_permutations = postprocess(
                        extracted_labels,
                        pos_tags,
                        top_order,
                        arg_perm=False,
                    )
                    assert len(extracted_labels_all_permutations) == 1
                    extracted_labels = extracted_labels_all_permutations[0]
                    self.__save_predicted_conll(
                        orig_conll,
                        extracted_labels,
                        f"{out_dir}/sen{sen_idx}_extracted_k{i}.conll"
                    )
                    wire_extractions[pp][sen_text].append(get_wire_extraction(
                        extracted_labels,
                        sen_text,
                        sen_id=int(sen_idx),
                        k=k,
                        score=score,
                        extractor=f"{predict_dir.split('/')[-3].split('dev_')[1]}_{chart_filter}_{pp}",
                        pred_res=pred_res,
                    ))
            for subdir, extractions in wire_extractions.items():
                wire_json = f"{chart_filter_dir}/{subdir}/sen{sen_idx}_wire.json"
                with open(wire_json, "w") as f:
                    json.dump(extractions, f, indent=4)

    def __get_preproc_input(self, sen_idx):
        preproc_dir = f"{self.preproc_dir}/{sen_idx}"

        parsed_doc_file = f"{preproc_dir}/parsed.conll"
        pos_tags = get_pos_tags(parsed_doc_file)

        top_order_file = f"{preproc_dir}/pos_edge_graph_top_order.json"
        top_order = json.load(open(top_order_file))

        orig_conll = f"{preproc_dir}/sen{sen_idx}.conll"
        sen_text = get_sen_txt(orig_conll)
        return sen_text, pos_tags, top_order, orig_conll

    def __save_predicted_conll(self, orig_conll, extracted_labels, extracted_conll):
        output = []
        with open(orig_conll) as f:
            lines = f.readlines()
        predicate = self.__get_predicate(extracted_labels)
        for line in lines:
            line = line.strip()
            fields = line.split("\t")
            output.append(fields[:2] + [predicate, extracted_labels.get(fields[0], "O"), str(1.0)])
        with open(extracted_conll, "w") as f:
            for line in output:
                f.write("\t".join(line))
                f.write("\n")

    @staticmethod
    def __get_predicate(extracted_labels):
        predicates = []
        for node, label in extracted_labels.items():
            if label == "P":
                predicates += [node]
        return f"[{' '.join([p for p in predicates])}]"


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)

    args = get_data_dir_and_config_args("Script to create wire jsons from predicted bolinas labels.")
    script = Predict(
        args.data_dir,
        args.config,
    )
    script.run()

