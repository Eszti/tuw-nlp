import os
from abc import abstractmethod

from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.sem.hrg.steps.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_rule import VoRule


class LoopOnSenDirs(Script):
    def __init__(self, description, log=True, config=None):
        super().__init__(description, log, config)
        self.in_dir = f"{self.data_dir}/{self.config['in_dir']}"

    def __get_range(self):
        first = self.first
        last = self.last
        sen_dirs = sorted([int(fn.split(".")[0]) for fn in os.listdir(f"{self.in_dir}")])
        if first is None or first < sen_dirs[0]:
            first = sen_dirs[0]
        if last is None or last > sen_dirs[-1]:
            last = sen_dirs[-1]
        return [n for n in sen_dirs if first <= n <= last]

    def _run_loop(self):
        for sen_idx in self.__get_range():
            if self.first_sen_to_proc is None:
                self.first_sen_to_proc = sen_idx
            print(f"\nProcessing folder {sen_idx}")
            sen_dir = f"{self.in_dir}/{str(sen_idx)}"
            self._do_for_sen(sen_idx, sen_dir)
            self.last_sen_to_proc = sen_idx

    @abstractmethod
    def _do_for_sen(self, sen_idx, sen_dir):
        raise NotImplemented

    def _load_grammar(self):
        grammar_file = f"{self._get_subdir('grammar', create=False)}/{self.config['grammar_file']}"

        with open(grammar_file) as f:
            self.grammar = Grammar.load_from_file(f, VoRule, reverse=False, nodelabels=True, logprob=True)

        rhs2_type = f"-to-{self.grammar.rhs2_type}" if self.grammar.rhs2_type else ''
        self._log(f"\nLoaded {self.grammar.rhs1_type}{rhs2_type} grammar with {len(self.grammar)} rules.")
