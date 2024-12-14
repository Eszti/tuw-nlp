import fileinput
import pickle

from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.parser import Parser


class Parse(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(description="Script to parse graph inputs and save parsed chars.", config=config)
        self.grammar = None
        self.parser = None

    def _before_loop(self):
        self._load_grammar()
        self.parser = Parser(self.grammar, max_steps=self.config.get("max_steps", 10000))

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
            partial=True,
        )

        for i, (chart, parse_logs) in enumerate(parse_generator):
            assert i == 0
            if "START" not in chart:
                self._log("No derivation found", sen_log_lines)
                continue
            else:
                self._log(parse_logs, sen_log_lines)
                with open(chart_file, "wb") as f:
                    pickle.dump(chart, f, -1)
        with open(sen_log_file, "w") as f:
            f.writelines(sen_log_lines)


if __name__ == "__main__":
    Parse().run()
