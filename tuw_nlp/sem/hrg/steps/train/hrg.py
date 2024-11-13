import os
from collections import Counter, defaultdict

from tuw_nlp.sem.hrg.common.script.loop_on_sen_dirs import LoopOnSenDirs


class Hrg(LoopOnSenDirs):
    def __init__(self, config=None):
        super().__init__(description="Script to merge hrg rules into one grammar file of a given size.", config=config)
        self.size = self.config.get("size", None)
        self.grammar = defaultdict(Counter)
        self.grammar_fn_name = f"{self.config_json.split('/')[-1].split('.json')[0]}"

    def _do_for_sen(self, sen_idx, sen_dir):
        filename = f"{sen_dir}/sen{sen_idx}.hrg"
        if os.path.exists(filename):
            with open(filename) as f:
                lines = f.readlines()
            for line in lines:
                rule = line.strip()
                nt = line.split(" ")[0]
                self.grammar[nt][rule] += 1

    def _after_loop(self):
        self.__cut_grammar()
        self.__add_weights()
        grammar_dir = self._get_subdir("grammar")
        with open(f"{grammar_dir}/{self.grammar_fn_name}.hrg", "w") as f:
            self.__write_rules(f, "weight")
        with open(f"{grammar_dir}/{self.grammar_fn_name}.stat", "w") as f:
            self.__write_rules(f, "cnt")
        self._log(f"\nUnique rules: {self.__get_total_number_of_rules()}")
        for nt, prods in self.grammar.items():
            self._log(f"{nt}: {len(prods)}\t({round(len(prods) / self.__get_total_number_of_rules(), 3)})")
        super()._after_loop()

    def __get_total_number_of_rules(self):
        all_prods = 0
        for nt, prods in self.grammar.items():
            all_prods += len(prods)
        return all_prods

    def __cut_grammar(self):
        if self.size:
            factor = self.size / self.__get_total_number_of_rules()
            new_grammar = defaultdict(Counter)
            for nt, prods in self.grammar.items():
                for prod, cnt in prods.most_common(n=int(round(factor * len(prods)))):
                    new_grammar[nt][prod] = cnt
            self.grammar = new_grammar

    def __add_weights(self):
        new_grammar = defaultdict(list)
        for nt, prods in self.grammar.items():
            for prod, cnt in prods.most_common():
                w = round(cnt / prods.total(), 2)
                if w < 0.01:
                    w = 0.01
                new_grammar[nt].append((prod, cnt, w))
        self.grammar = new_grammar

    def __write_rules(self, f, numeric_info=None):
        for (prod, cnt, w) in self.grammar["S"]:
            if not numeric_info:
                f.write(f"{prod}\n")
            elif numeric_info == "cnt":
                f.write(f"{prod}\t{cnt}\n")
            elif numeric_info == "weight":
                f.write(f"{prod}\t{w}\n")
        for nt, prods in self.grammar.items():
            if nt == "S":
                continue
            for (prod, cnt, w) in prods:
                if not numeric_info:
                    f.write(f"{prod}\n")
                elif numeric_info == "cnt":
                    f.write(f"{prod}\t{cnt}\n")
                elif numeric_info == "weight":
                    f.write(f"{prod}\t{w}\n")


if __name__ == "__main__":
    Hrg().run()
