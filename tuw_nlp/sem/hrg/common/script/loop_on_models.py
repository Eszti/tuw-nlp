from abc import abstractmethod

from tuw_nlp.sem.hrg.common.script.script import Script


class LoopOnModels(Script):
    def __init__(self, data_dir, config_json, log=False):
        super().__init__(data_dir, config_json, log)
        self.in_dir = f"{self.data_dir}/{self.config['in_dir']}"
        self.models = self.config["models"]
        self.report = ""
        self.report_dir = self._get_subdir("reports")

    def _run_loop(self):
        for model in self.models:
            if model.get("ignore") and model["ignore"]:
                continue
            print(f"Processing model {model}")
            self._do_for_model(model)

    @abstractmethod
    def _do_for_model(self, model):
        raise NotImplemented
