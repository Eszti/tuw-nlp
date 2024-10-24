import os
from abc import abstractmethod

from tuw_nlp.sem.hrg.common.script.script import Script


class LoopScriptOnPreprocessed(Script):
    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json)
        self.in_dir = f"{self.data_dir}/{self.config['preproc_dir']}"
        self.out_dir = f"{self.data_dir}/{self.config['out_dir']}"
        self.first_sen_to_proc = None
        self.last_sen_to_proc = None

    def __get_range(self):
        first = self.config.get("first", None)
        last = self.config.get("last", None)
        sen_dirs = sorted([int(fn.split(".")[0]) for fn in os.listdir(f"{self.data_dir}/{self.config['preproc_dir']}")])
        if first is None or first < sen_dirs[0]:
            first = sen_dirs[0]
        if last is None or last > sen_dirs[-1]:
            last = sen_dirs[-1]
        return [n for n in sen_dirs if first <= n <= last]

    def run(self):
        self.before_loop()
        for sen_idx in self.__get_range():
            if self.first_sen_to_proc is None:
                self.first_sen_to_proc = sen_idx
            print(f"\nProcessing sen {sen_idx}\n")
            preproc_dir = f"{self.in_dir}/{str(sen_idx)}/preproc"
            self.run_loop(sen_idx, preproc_dir)
            self.last_sen_to_proc = sen_idx
        self.after_loop()

    @abstractmethod
    def before_loop(self):
        raise NotImplemented

    @abstractmethod
    def run_loop(self, sen_idx, preproc_dir):
        raise NotImplemented

    @abstractmethod
    def after_loop(self):
        raise NotImplemented
