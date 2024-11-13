import json
import os
from abc import abstractmethod

from tuw_nlp.sem.hrg.common.script.script import Script


class LoopOnModels(Script):
    def __init__(self, description, log=False, config=None):
        super().__init__(description, log, config)
        if "in_dir" in self.config:
            self.in_dir = f"{self.data_dir}/{self.config['in_dir']}"
        self.models = self.config["models"]
        self.report = ""
        if "gold_fn" in self.config:
            gold_path = f"{os.path.dirname(self.pipeline_dir)}/data/{self.config['gold_fn']}"
            self.gold = json.load(open(gold_path))
        self.last = self.config.get("last", None)

    def _run_loop(self):
        for model in self.models:
            if model.get("ignore") and model["ignore"]:
                continue
            print(f"Processing model {model}")
            self._do_for_model(model)

    @abstractmethod
    def _do_for_model(self, model):
        raise NotImplemented

    def _get_merged_jsons(self, in_dir, chart_filter, pp, only_all=False):
        in_dir = self._add_filter_and_postprocess(in_dir, chart_filter, pp)
        files = [i for i in os.listdir(in_dir) if i.endswith(".json")]
        if only_all:
            files = [i for i in files if i.endswith("_all.json")]
            assert len(files) == 1
        else:
            k_files = [i for i in files if i.split("_")[-1].startswith("k")]
            if k_files:
                files = sorted(k_files, key=lambda x: int(x.split('.')[0].split("_")[-1].split("k")[-1]))
        return [f"{in_dir}/{f}" for f in files]

    @staticmethod
    def _get_all_sen_dirs(in_dir):
        sen_dirs = sorted([int(fn.split('.')[0]) for fn in os.listdir(in_dir)])
        return [f"{in_dir}/{sen_dir}" for sen_dir in sen_dirs]
