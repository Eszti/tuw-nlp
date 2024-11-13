import os.path
import fileinput
import pickle

from tuw_nlp.sem.hrg.common.io import log_to_console_and_log_lines
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.steps.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_rule import VoRule


class Parse(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(description="Script to parse graph inputs and save parsed chars.", config=config)
        self.grammar = None
        self.parser = None

    def _before_loop(self):
        self._load_grammar()
        self.parser = Parser(self.grammar)

    def _load_grammar(self):
        logprob = True
        nodelabels = True
        backward = False

        grammar_file = f"{self._get_subdir('grammar', create=False)}/{self.config['grammar_file']}"

        with open(grammar_file) as f:
            self.grammar = Grammar.load_from_file(f, VoRule, backward, nodelabels=nodelabels, logprob=logprob)

        rhs2_type = f"-to-{self.grammar.rhs2_type}" if self.grammar.rhs2_type else ''
        self._log(f"\nLoaded {self.grammar.rhs1_type}{rhs2_type} grammar with {len(self.grammar)} rules.")

    def _do_for_sen(self, sen_idx, sen_dir):
        bolinas_dir = self._get_subdir("bolinas", parent_dir=f"{self.out_dir}/{str(sen_idx)}")
        self._parse_sen(
            graph_file=f"{sen_dir}/pos_edge.graph",
            chart_file=f"{bolinas_dir}/sen{str(sen_idx)}_chart.pickle",
            sen_log_file=f"{bolinas_dir}/sen{str(sen_idx)}_parse.log",
        )

    def _parse_sen(self, graph_file, chart_file, sen_log_file):
        sen_log_lines = []
        parse_generator = self.parser.parse_graphs(
            (Hgraph.from_string(x) for x in fileinput.input(graph_file)),
            sen_log_lines,
            partial=True,
            max_steps=self.config.get("max_steps", 10000)
        )

        for i, chart in enumerate(parse_generator):
            assert i == 0
            if "START" not in chart:
                self._log("No derivation found", sen_log_lines)
                continue
            else:
                self._log(f"Chart len: {len(chart)}", sen_log_lines)
                with open(chart_file, "wb") as f:
                    pickle.dump(chart, f, -1)
        with open(sen_log_file, "w") as f:
            f.writelines(sen_log_lines)


if __name__ == "__main__":
    Parse().run()
