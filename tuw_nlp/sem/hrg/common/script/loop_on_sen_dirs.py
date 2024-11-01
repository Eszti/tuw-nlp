import os
from abc import abstractmethod

from tuw_nlp.sem.hrg.common.script.script import Script


class LoopOnSenDirs(Script):
    def __init__(self, data_dir, config_json, log=True):
        super().__init__(data_dir, config_json, log)
        self.in_dir = f"{self.data_dir}/{self.config['in_dir']}"

    def __get_range(self):
        first = self.config.get("first", None)
        last = self.config.get("last", None)
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
