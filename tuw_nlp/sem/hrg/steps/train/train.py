import json
import logging

from tuw_nlp.graph.graph import Graph
from tuw_nlp.sem.hrg.common.script.loop_on_triplets import LoopOnTriplets
from tuw_nlp.sem.hrg.steps.bolinas.common.exceptions import ParseTooLongException, CkyTooLongException, \
    NotAllNodesCoveredException
from tuw_nlp.sem.hrg.steps.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_rule import VoRule
from tuw_nlp.sem.hrg.steps.bolinas.validate.check_membership import check_membership
from tuw_nlp.sem.hrg.steps.train.rule_generation.per_word import get_rules_per_word


class Train(LoopOnTriplets):
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

    def _do_for_triplet(self, sen_dir, triplet_idx, triplet_graph_str, triplet):
        hrg_dir = self._get_subdir(str(triplet_idx), self.out_dir)
        triplet_log = open(f"{hrg_dir}/sen{triplet_idx}.log", "w")
        triplet_graph = Graph.from_bolinas(triplet_graph_str)

        initial_rule = ""
        rules = []
        if self.method == "per_word":
            initial_rule, rules = get_rules_per_word(triplet_graph, triplet)
        # Todo: per_arg
        if initial_rule is None:
            assert len(rules) == 0
            self.no_rule.append(triplet_idx)
        else:
            grammar_lines = [f"{initial_rule}"]
            for rule in sorted(rules):
                grammar_lines.append(f"{rule}")
            triplet_log.write(f"Grammar length: {len(grammar_lines)}\n")
            grammar = Grammar.load_from_file(grammar_lines, VoRule, nodelabels=True, logprob=True)
            parser = Parser(grammar, stop_at_first=True, permutations=False)

            try:
                log_from_validator, used_rules = check_membership(parser, triplet_graph_str)
                triplet_log.writelines(log_from_validator)
                if used_rules is None:
                    self.not_validated.append(triplet_idx)
                if len(used_rules.keys()) != len(grammar):
                    triplet_log.writelines(f"\nNot all rules are used: {len(used_rules.keys())} of {len(grammar)}\n")
                    self.not_all_rules_used.append(triplet_idx)
                with open(f"{hrg_dir}/sen{triplet_idx}.hrg", "w") as f:
                    f.writelines(grammar_lines)
            except ParseTooLongException as e:
                self.parse_did_not_finish.append(triplet_idx)
                triplet_log.write(e.print_message())
            except CkyTooLongException as e:
                self.cky_did_not_finish.append(triplet_idx)
                triplet_log.write(e.print_message())
            except NotAllNodesCoveredException as e:
                self.not_all_nodes_covered.append(triplet_idx)
                triplet_log.write(e.print_message())

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
