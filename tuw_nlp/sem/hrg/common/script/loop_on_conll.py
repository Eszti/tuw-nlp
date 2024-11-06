import os
from abc import abstractmethod

from tuw_nlp.sem.hrg.common.conll import get_sen_text
from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.text.utils import gen_tsv_sens


class LoopOnConll(Script):
    def __init__(self, description, log=True):
        super().__init__(description, log)
        self.conll_file = f"{self.data_dir}/{self.config['in_file']}"

    def _run_loop(self):
        first = self.config.get("first", None)
        last = self.config.get("last", None)
        last_sen_txt = ""
        sen_dir = ""

        for sen_idx, sen in enumerate(gen_tsv_sens(open(self.conll_file))):
            if self.first_sen_to_proc is None:
                self.first_sen_to_proc = sen_idx
            if first is not None and sen_idx < first:
                continue
            if last is not None and last < sen_idx:
                break

            print(f"Processing sen {sen_idx}")
            sen_txt = get_sen_text(sen)

            if self.out_dir:
                if sen_txt != last_sen_txt:
                    sen_dir = f"{self.out_dir}/{sen_idx}"
                    if not os.path.exists(sen_dir):
                        os.makedirs(sen_dir)
                else:
                    assert os.path.exists(sen_dir)

            self._do_for_sen(sen_idx, sen, sen_txt, last_sen_txt, sen_dir)

            last_sen_txt = sen_txt
            self.last_sen_to_proc = sen_idx

    @abstractmethod
    def _do_for_sen(self, sen_idx, sen, sen_txt, last_sen_txt, sen_dir):
        raise NotImplemented
