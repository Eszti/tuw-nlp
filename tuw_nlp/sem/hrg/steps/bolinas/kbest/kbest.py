import json
import math
import os.path
import pickle

from collections import OrderedDict
from copy import copy

from tuw_nlp.sem.hrg.common.conll import ConllSen
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.steps.bolinas.common.exceptions import DerivationException
from tuw_nlp.sem.hrg.steps.bolinas.common.oie import get_labels, extract_for_kth_derivation
from tuw_nlp.sem.hrg.steps.bolinas.kbest.filter.pr_filter import filter_for_pr
from tuw_nlp.sem.hrg.steps.bolinas.kbest.filter.size_filter import filter_for_size


def get_k_best_unique_derivation(chart, k):
    kbest_unique_nodes = set()
    kbest_unique_derivations = []
    for score, derivation in chart:
        final_item = derivation[1]["START"][0]
        nodes = sorted(list(final_item.nodeset), key=lambda node: int(node[1:]))
        nodes_str = " ".join(nodes)
        if nodes_str not in kbest_unique_nodes:
            kbest_unique_nodes.add(nodes_str)
            kbest_unique_derivations.append((score, derivation))
        if len(kbest_unique_derivations) >= k:
            break
    assert len(kbest_unique_derivations) == len(kbest_unique_nodes)
    if len(kbest_unique_derivations) < k:
        print(f"Found only {len(kbest_unique_derivations)} derivations.")
    return kbest_unique_derivations


def save_output(outputs):
    for (fn, lines) in outputs:
        if lines:
            with open(fn, "w") as f:
                f.writelines(lines)


def get_gold_labels(preproc_dir):
    gold_labels = []
    files = [fn for fn in os.listdir(preproc_dir) if fn.endswith("_triplet.txt")]
    for fn in files:
        with open(f"{preproc_dir}/{fn}") as f:
            gold_labels.append(json.load(f))
    return gold_labels


class KBest(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(description="Script to search k best derivations in parsed charts.", config=config)
        self.logprob = True
        self.score_disorder_collector = {}

    def _before_loop(self):
        pass

    def _do_for_sen(self, sen_idx, sen_dir):
        bolinas_dir = f"{self.out_dir}/{str(sen_idx)}/bolinas"
        chart_file = f"{bolinas_dir}/sen{sen_idx}_chart.pickle"
        if not os.path.exists(chart_file):
            return

        with open(chart_file, "rb") as f:
            chart = pickle.load(f)

        if "START" not in chart:
            print("No derivation found")
            return

        gold_labels = get_gold_labels(sen_dir)
        top_order = json.load(open(
            f"{sen_dir}/pos_edge_graph_top_order.json"
        ))
        pos_tags = ConllSen(sen_dir).pos_tags()

        for name, c in sorted(self.config["filters"].items()):
            if c.get("ignore", False):
                continue
            print(f"Processing {name}")
            matches_lines = []
            labels_lines = []
            rules_lines = []
            sen_log_lines = []

            filtered_chart = copy(chart)
            sen_log_lines.append(f"Chart 'START' length: {len(filtered_chart['START'])}\n")
            if "chart_filter" in c:
                chart_filter = c["chart_filter"]
                assert chart_filter in ["basic", "max"]
                filtered_chart = filter_for_size(chart, chart_filter)
            sen_log_lines.append(f"Chart 'START' length after size filter: {len(filtered_chart['START'])}\n")

            derivations = filtered_chart.derivations("START")

            assert ("k" in c and "pr_metric" not in c) or ("k" not in c and "pr_metric" in c)

            labels_with_arg_idx = []
            if "k" in c:
                k_best_unique_derivations = get_k_best_unique_derivation(derivations, c["k"])
            elif "pr_metric" in c:
                metric = c["pr_metric"]
                assert metric in ["prec", "rec", "f1"]
                k_best_unique_derivations, labels_with_arg_idx = filter_for_pr(
                    derivations,
                    gold_labels,
                    metric,
                    pos_tags,
                    top_order,
                    self.config["arg_permutation"],
                )
                if len(k_best_unique_derivations) == 0:
                    sen_log_lines.append("No matches\n")
            else:
                print("Neither 'k' nor 'pr_metric' is set")
                continue

            last_score = None
            score_disorder = {}
            for i, (score, derivation) in enumerate(k_best_unique_derivations):
                ki = i + 1
                if "k" in c:
                    n_score = score if self.logprob else math.exp(score)
                else:
                    n_score = score

                new_score = score
                if last_score:
                    if new_score > last_score:
                        order_str = "%d-%d" % (ki - 1, ki)
                        score_disorder[order_str] = (last_score, new_score)
                last_score = new_score

                try:
                    shifted_derivation, used_rules, matched_nodes = extract_for_kth_derivation(
                        derivation,
                        n_score,
                        ki,
                    )
                    matches_lines.append(shifted_derivation)
                    rules_lines.append(f"{used_rules}\n")
                    sen_log_lines.append(matched_nodes)
                    if "pr_metric" in c:
                        labels = labels_with_arg_idx[i]
                    else:
                        labels = get_labels(derivation)
                    labels_lines.append(
                        f"{json.dumps(OrderedDict(sorted(labels.items(), key=lambda x: int(x[0]))))};{n_score}\n")
                except DerivationException as e:
                    print("Could not construct derivation: '%s'. Skipping." % e)

            for i, val in score_disorder.items():
                sen_log_lines.append("%s: %g / %g\n" % (i, val[0], val[1]))
            self.score_disorder_collector[sen_idx] = (len(score_disorder.items()), len(k_best_unique_derivations))

            out_dir = os.path.join(bolinas_dir, name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_output(
                [
                    (f"{out_dir}/sen{sen_idx}_matches.graph", matches_lines),
                    (f"{out_dir}/sen{sen_idx}_predicted_labels.txt", labels_lines),
                    (f"{out_dir}/sen{sen_idx}_derivation.txt", rules_lines),
                    (f"{out_dir}/sen{sen_idx}.log", sen_log_lines),
                ]
            )

    def _after_loop(self):
        num_sem = len(self.score_disorder_collector.keys())
        self._log(f"\nNumber of sentences: {num_sem}")
        sum_score_disorder = sum([val[0] for val in self.score_disorder_collector.values()])
        self._log(f"Sum of score disorders: {sum_score_disorder}")
        self._log(f"Average score disorders: {round(sum_score_disorder / float(num_sem), 2)}")
        super()._after_loop()


if __name__ == "__main__":
    KBest().run()
