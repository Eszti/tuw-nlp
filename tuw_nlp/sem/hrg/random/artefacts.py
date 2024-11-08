import json
from collections import defaultdict, Counter

from tuw_nlp.sem.hrg.common.script.loop_on_conll import LoopOnConll


def get_labels_str(sen):
    return "_".join([line[-1].split("-")[0] for line in sen])


class ArtefactsExtractor(LoopOnConll):

    def __init__(self, config=None):
        super().__init__(description="Script to extract artefacts of the given dataset.", config=config)
        self.artefact_dir = f"{self._get_subdir('artefacts')}"
        self.sequences = defaultdict(list)
        self.nr_ex_stat = Counter()
        self.cnt = 1

    def _do_for_sen(self, sen_idx, sen, sen_txt, last_sen_txt, sen_dir):
        self.sequences[len(sen)].append(get_labels_str(sen))
        sen_txt = " ".join([word[1] for word in sen])
        if last_sen_txt == sen_txt:
            self.cnt += 1
        elif last_sen_txt != "":
            self.nr_ex_stat[self.cnt] += 1
            self.cnt = 1

    def _after_loop(self):
        with open(f"{self.artefact_dir}/{self.config_name}_sequences.json", "w") as f:
            json.dump({key: v for key, v in sorted(self.sequences.items())}, f)
        with open(f"{self.artefact_dir}/{self.config_name}_k_dist.json", "w") as f:
            json.dump({key: v for key, v in sorted(self.nr_ex_stat.items())}, f)
        super()._after_loop()


if __name__ == "__main__":
    ArtefactsExtractor().run()

