from abc import ABC

from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.sem.hrg.preproc.preproc import Preproc
from tuw_nlp.sem.hrg.train.hrg import Hrg
from tuw_nlp.sem.hrg.train.train import Train


class Pipeline(Script, ABC):
    def __init__(self, description, log=True, config=None):
        super().__init__(description, log, config)
        self.pipeline_name = self.config["name"]
        self.steps = self.config["steps"]
        self.name_to_class = {
            "preproc": Preproc,
            "train": Train,
            "hrg": Hrg,
        }
