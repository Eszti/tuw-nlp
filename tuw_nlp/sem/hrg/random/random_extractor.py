import json
import os
import random
from collections import defaultdict

from tuw_nlp.sem.hrg.common.conll import ConllSen
from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction
from tuw_nlp.sem.hrg.postproc.postproc import add_arg_idx

random.seed(10)


class RandomExtractor(LoopOnSenDirs):

    def __init__(self, description):
        super().__init__(description, log=True)
        self.artefact_dir = self._get_subdir("artefacts")
        self.artefact_prefix = self.config["artefact_prefix"]
        self.k_max = self.config.get("k_max", 10)
        self.models = self.config["models"]
        self.non_verbs = defaultdict(list)

    def _before_loop(self):
        self.__read_k_dist()
        self.__read_sequences()

    def __read_k_dist(self):
        with open(f"{self.artefact_dir}/{self.artefact_prefix}_k_dist.json") as f:
            self.k_dist = json.load(f)
        to_del = []
        for k, v in self.k_dist.items():
            if int(k) > self.k_max:
                to_del.append(k)
        for k in to_del:
            del self.k_dist[k]

    def __read_sequences(self):
        self.sequences = json.load(open(f"{self.artefact_dir}/{self.artefact_prefix}_sequences.json"))
        self._log(f"sentence lengths: {len(self.sequences)}")
        self._log(f"{sorted(self.sequences)}")

    def _do_for_sen(self, sen_idx, sen_dir):
        out_sen_dir = f"{self.out_dir}/{str(sen_idx)}"
        if not os.path.exists(out_sen_dir):
            os.makedirs(out_sen_dir)

        conll_sen = ConllSen(sen_dir)
        sen_len = conll_sen.len()
        sen_txt = conll_sen.sen_text()
        pos_tags = conll_sen.pos_tags()

        sen_len_for_stat = str(sen_len)
        if sen_len_for_stat not in self.sequences:
            while sen_len_for_stat not in self.sequences:
                sen_len_for_stat = str(int(sen_len_for_stat) - 1)

        nr_seq = len(self.sequences[sen_len_for_stat])
        nr_ex = random.choices(list(self.k_dist.keys()), list(self.k_dist.values()))
        nr_to_gen = int(nr_ex[0])

        extracted = defaultdict(lambda: defaultdict(list))
        used_rnd_indexes = set()
        for i in range(nr_to_gen):
            rnd_idx = random.randrange(nr_seq)
            while rnd_idx in used_rnd_indexes:
                rnd_idx = random.randrange(nr_seq)
            used_rnd_indexes.add(rnd_idx)

            pred_seq = self.sequences[sen_len_for_stat][rnd_idx]
            if sen_len > len(pred_seq):
                pred_seq += "_"
                pred_seq += "_".join(["O"] * (sen_len - len(pred_seq)))

            extracted_labels = {str(i + 1): l for i, l in enumerate(pred_seq.split("_"))}
            self.__ensure_verb_pred(extracted_labels, sen_idx, pos_tags)
            for model in sorted(self.models):
                model_labels = extracted_labels
                if model == "boa":
                    model_labels = {i: "A" if l.startswith("A") else l for i, l in extracted_labels.items()}
                    add_arg_idx(model_labels, sen_len, arg_perm=False)
                extracted[model][sen_txt].append(
                    get_wire_extraction(
                        model_labels,
                        sen_txt,
                        i + 1,
                        sen_idx,
                        extractor=f"random_{model}",
                        )
                )
        for model, ex in extracted.items():
            model_dir = f"{out_sen_dir}/{model}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            with open(f"{model_dir}/sen{sen_idx}_wire.json", "w") as f:
                json.dump(ex, f, indent=4)

    def __ensure_verb_pred(self, extracted_labels, sen_idx, pos_tags):
        p_idx_l = [i for i, label in extracted_labels.items() if label == "P"]
        verbs = [str((i + 1)) for i, t in enumerate(pos_tags) if t == "VERB"]
        if verbs:
            for p_idx in p_idx_l:
                if pos_tags[int(p_idx) - 1] != "VERB":
                    new_p_idx = verbs[random.randrange(0, len(verbs))]
                    extracted_labels[new_p_idx] = "P"
                    extracted_labels[p_idx] = "O"
        else:
            self.non_verbs[sen_idx].append((p_idx_l, pos_tags))

    def _after_loop(self):
        self._log(self.non_verbs)
        super()._after_loop()


if __name__ == "__main__":
    RandomExtractor("Script to generate random extractions.").run()
