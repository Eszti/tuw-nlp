import json
import logging
import os

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.common.triplet import Triplet
from tuw_nlp.sem.hrg.steps.bolinas.common.exceptions import ParseTooLongException, CkyTooLongException
from tuw_nlp.sem.hrg.steps.bolinas.validate.validate import check_if_graph_accepted_by_hrg
from tuw_nlp.sem.hrg.steps.train.rule_generation.per_word import get_rules_per_word


class Train(LoopOnSenDirs):
    def __init__(self, config=None):
        super().__init__(description="Script to create hrg rules on preprocessed train data.", config=config)
        self.method = self.config["method"]
        self.out_dir += f"_{self.method}"
        self.no_rule = []
        self.not_validated = []
        self.not_all_rules_used = []
        self.not_all_nodes_covered = []
        self.parse_did_not_finish = []
        self.cky_did_not_finish = []

    def _do_for_sen(self, sen_idx, sen_dir):
        graph_files = sorted([f"{sen_dir}/{fn}" for fn in os.listdir(sen_dir) if fn.endswith("_triplet.graph")])
        for graph_file in graph_files:
            exact_sen_idx = int(graph_file.split("/")[-1].split("_triplet.graph")[0].split("sen")[-1])
            hrg_dir = f"{self.out_dir}/{str(exact_sen_idx)}"
            if not os.path.exists(hrg_dir):
                os.makedirs(hrg_dir)
            log = open(f"{hrg_dir}/sen{exact_sen_idx}.log", "w")
            print(f"\nProcessing sen {exact_sen_idx}")

            with open(graph_file) as f:
                lines = f.readlines()
                assert len(lines) == 1
                graph_str = lines[0].strip()
            triplet_graph = Graph.from_bolinas(graph_str)
            triplet = Triplet.from_file(f"{sen_dir}/sen{exact_sen_idx}_triplet.txt")

            initial_rule = ""
            rules = []
            if self.method == "per_word":
                initial_rule, rules = get_rules_per_word(triplet_graph, triplet)
            # Todo: per_arg
            if initial_rule is None:
                assert len(rules) == 0
                self.no_rule.append(exact_sen_idx)
            else:
                grammar_lines = [f"{initial_rule}"]
                for rule in sorted(rules):
                    grammar_lines.append(f"{rule}")
                print(f"Grammar length: {len(grammar_lines)}")

                try:
                    log_from_validator, accepted, all_rules_used, all_nodes_covered = check_if_graph_accepted_by_hrg(
                        grammar_lines,
                        graph_str
                    )
                    if not accepted:
                        self.not_validated.append(exact_sen_idx)
                    if not all_rules_used:
                        self.not_all_rules_used.append(exact_sen_idx)
                    if not all_nodes_covered:
                        self.not_all_nodes_covered.append(exact_sen_idx)
                    log.writelines(log_from_validator)

                    with open(f"{hrg_dir}/sen{exact_sen_idx}.hrg", "w") as f:
                        f.writelines(grammar_lines)
                except ParseTooLongException as e:
                    self.parse_did_not_finish.append(exact_sen_idx)
                    log.write(
                        f"Parse did not finish:\n"
                        f"- steps: {e.steps}\n"
                        f"- queue len: {e.queue_len}\n"
                        f"- attempted len:{e.attempted_len}\n"
                    )
                except CkyTooLongException as e:
                    self.cky_did_not_finish.append(exact_sen_idx)
                    log.write(
                        f"Cky conversion did not finish:\n"
                        f"- steps: {e.steps}\n"
                    )

    def _after_loop(self):
        self._log(
            f"\nNumber of no rules: {len(self.no_rule)}\n"
            f"{json.dumps(self.no_rule)}"
            f"\nNumber of not validated: {len(self.not_validated)}\n"
            f"{json.dumps(self.not_validated)}"
            f"\nNumber of not all rules used: {len(self.not_all_rules_used)}\n"
            f"{json.dumps(self.not_all_rules_used)}"
            f"\nNumber of not all nodes covered: {len(self.not_all_nodes_covered)}\n"
            f"{json.dumps(self.not_all_nodes_covered)}"
            f"\nNumber of parse did not finish: {len(self.parse_did_not_finish)}\n"
            f"{json.dumps(self.parse_did_not_finish)}"
            f"\nNumber of cky conversion did not finish: {len(self.cky_did_not_finish)}\n"
            f"{json.dumps(self.cky_did_not_finish)}",
        )
        super()._after_loop()


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)
    Train().run()
