import json
import logging

from tuw_nlp.sem.hrg.common.script.loop_on_triplets import LoopOnTriplets
from tuw_nlp.sem.hrg.steps.bolinas.common.exceptions import ParseTooLongException, CkyTooLongException
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.steps.bolinas.validate.check_membership import check_membership


class Validate(LoopOnTriplets):
    def __init__(self, config=None):
        super().__init__(description="Script to check whether dev triplets are accepted by the grammar.", config=config)
        self.not_validated = []
        self.parse_did_not_finish = []
        self.cky_did_not_finish = []
        self.not_all_nodes_covered = []
        self.validated = []
        self.all_dev_triplets = 0

    def _before_loop(self):
        self._load_grammar()
        self.parser = Parser(self.grammar, stop_at_first=True)

    def _do_for_triplet(self, sen_dir, triplet_idx, triplet_graph_str, triplet):
        self.all_dev_triplets += 1
        hrg_dir = self._get_subdir(str(triplet_idx), self.out_dir)
        triplet_log = open(f"{hrg_dir}/sen{triplet_idx}.log", "w")
        try:
            log_from_validator, used_rules = check_membership(self.parser, triplet_graph_str)
            triplet_log.writelines(log_from_validator)
            if used_rules is None:
                self.not_validated.append(triplet_idx)
            else:
                self.validated.append(triplet_idx)
        except ParseTooLongException as e:
            self.parse_did_not_finish.append(triplet_idx)
            triplet_log.write(e.print_message())
        except CkyTooLongException as e:
            self.cky_did_not_finish.append(triplet_idx)
            triplet_log.write(e.print_message())

    def _after_loop(self):
        val_ratio = float(len(self.validated)) / self.all_dev_triplets
        self._log(
            f"\nValidation ratio: {round(val_ratio, 2)} ({len(self.validated)} / {self.all_dev_triplets})\n"
            f"\nNumber of validated: {len(self.validated)}\n"
            f"{json.dumps(self.validated)}"
            f"\nNumber of not validated: {len(self.not_validated)}\n"
            f"{json.dumps(self.not_validated)}"
            f"\nNumber of parse did not finish: {len(self.parse_did_not_finish)}\n"
            f"{json.dumps(self.parse_did_not_finish)}"
            f"\nNumber of cky conversion did not finish: {len(self.cky_did_not_finish)}\n"
            f"{json.dumps(self.cky_did_not_finish)}",
            print_to_std=True,
        )
        super()._after_loop()


if __name__ == "__main__":
    logging.getLogger('penman').setLevel(logging.ERROR)
    Validate().run()
