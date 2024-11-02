import os.path
import fileinput
import pickle

from tuw_nlp.sem.hrg.bolinas.common.grammar import Grammar
from tuw_nlp.sem.hrg.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.bolinas.parser_basic.parser import Parser
from tuw_nlp.sem.hrg.bolinas.parser_basic.vo_rule import VoRule
from tuw_nlp.sem.hrg.common.io import log_to_console_and_log_lines, get_data_dir_and_config_args
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs


class Parse(LoopOnSenDirs):

    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json)
        self.grammar = None
        self.parser = None

    def _before_loop(self):
        self._load_grammar()
        self.parser = Parser(self.grammar)

    def _load_grammar(self):
        logprob = True
        nodelabels = True
        backward = False

        grammar_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}//train/grammar/"
        grammar_file = f"{grammar_dir}/{self.config['grammar_file']}"

        with open(grammar_file) as f:
            self.grammar = Grammar.load_from_file(f, VoRule, backward, nodelabels=nodelabels, logprob=logprob)

        rhs2_type = f"-to-{self.grammar.rhs2_type}" if self.grammar.rhs2_type else ''
        log_to_console_and_log_lines(
            f"\nLoaded {self.grammar.rhs1_type}{rhs2_type} grammar with {len(self.grammar)} rules.",
            self.log_lines
        )

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
    args = get_data_dir_and_config_args("Script to parse graph inputs and save parsed chars.")
    script = Parse(
        args.data_dir,
        args.config,
    )
    script.run()
