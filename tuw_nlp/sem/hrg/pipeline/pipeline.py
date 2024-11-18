from datetime import datetime

from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.sem.hrg.steps.bolinas.kbest.kbest import KBest
from tuw_nlp.sem.hrg.steps.bolinas.parse.parse import Parse
from tuw_nlp.sem.hrg.steps.eval.eval import Eval
from tuw_nlp.sem.hrg.steps.predict.merge import Merge
from tuw_nlp.sem.hrg.steps.predict.predict import Predict
from tuw_nlp.sem.hrg.steps.preproc.preproc import Preproc
from tuw_nlp.sem.hrg.steps.random.artefacts import Artefacts
from tuw_nlp.sem.hrg.steps.random.random_extractor import Random
from tuw_nlp.sem.hrg.steps.stat.run_all_stat import Stat
from tuw_nlp.sem.hrg.steps.train.hrg import Hrg
from tuw_nlp.sem.hrg.steps.train.train import Train


class Pipeline(Script):
    def __init__(self, log=True, config=None):
        super().__init__("Script to run a pipeline.", log, config)
        self.steps = self.config["steps"]
        self.name_to_class = {
            "preproc": Preproc,
            "train": Train,
            "hrg": Hrg,
            "artefacts": Artefacts,
            "random": Random,
            "parse": Parse,
            "kbest": KBest,
            "predict": Predict,
            "merge": Merge,
            "eval": Eval,
            "stat": Stat,
        }

    def _run_loop(self):
        for step in self.steps:
            step_name = step['step_name']
            script_name = step['script_name']
            self._log(f"Processing step {step_name}: {datetime.now()}")
            step_class = self.name_to_class[script_name]
            config = f"{self.pipeline_dir}/config/{step['config']}"
            step = step_class(config=config)
            if self.first is not None:
                step.first = self.first
            if self.last is not None:
                step.last = self.last
            step.run()


if __name__ == "__main__":
    Pipeline().run()
