import os
from abc import abstractmethod

from tuw_nlp.sem.hrg.common.conll import get_sen_from_conll_sen
from tuw_nlp.sem.hrg.common.io import create_sen_dir
from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.text.utils import gen_tsv_sens


class LoopScriptOnPreprocessed(Script):
    def __init__(self, data_dir, config_json, log_time=False):
        super().__init__(data_dir, config_json, log_time)
        self.in_dir = f"{self.data_dir}/{self.config['preproc_dir']}"
        self.out_dir = f"{self.data_dir}/{self.config['out_dir']}"

    def __get_range(self):
        first = self.config.get("first", None)
        last = self.config.get("last", None)
        sen_dirs = sorted([int(fn.split(".")[0]) for fn in os.listdir(f"{self.data_dir}/{self.config['preproc_dir']}")])
        if first is None or first < sen_dirs[0]:
            first = sen_dirs[0]
        if last is None or last > sen_dirs[-1]:
            last = sen_dirs[-1]
        return [n for n in sen_dirs if first <= n <= last]

    def _run_loop(self):
        for sen_idx in self.__get_range():
            if self.first_sen_to_proc is None:
                self.first_sen_to_proc = sen_idx
            print(f"\nProcessing preproc folder {sen_idx}")
            preproc_dir = f"{self.in_dir}/{str(sen_idx)}/preproc"
            self._do_for_sen(sen_idx, preproc_dir)
            self.last_sen_to_proc = sen_idx

    @abstractmethod
    def _do_for_sen(self, sen_idx, preproc_dir):
        raise NotImplemented


class LoopScriptOnConll(Script):
    def __init__(self, data_dir, config_json, log_time=True):
        super().__init__(data_dir, config_json, log_time)
        self.conll_file = f"{self.data_dir}/{self.config['in_file']}"
        self.out_dir = f"{self.data_dir}/{self.config['out_dir']}"

    def _run_loop(self):
        first = self.config.get("first", None)
        last = self.config.get("last", None)
        last_sen_txt = ""
        preproc_dir = ""

        for sen_idx, sen in enumerate(gen_tsv_sens(open(self.conll_file))):
            if self.first_sen_to_proc is None:
                self.first_sen_to_proc = sen_idx
            if first is not None and sen_idx < first:
                continue
            if last is not None and last < sen_idx:
                break

            print(f"Processing sen {sen_idx}")
            sen_txt = get_sen_from_conll_sen(sen)

            if sen_txt != last_sen_txt:
                sen_dir = create_sen_dir(f"{self.out_dir}", sen_idx)
                preproc_dir = f"{sen_dir}/preproc"
                if not os.path.exists(preproc_dir):
                    os.makedirs(preproc_dir)
            else:
                assert os.path.exists(preproc_dir)

            self._do_for_sen(sen_idx, sen_txt, sen, last_sen_txt, preproc_dir)

            last_sen_txt = sen_txt
            self.last_sen_to_proc = sen_idx

    @abstractmethod
    def _do_for_sen(self, sen_idx, sen_txt, sen, last_sen_txt, preproc_dir):
        raise NotImplemented
