import os


class ConllSen:
    def __init__(self, sen_dir):
        orig_fns = [f"{sen_dir}/{fn}" for fn in os.listdir(sen_dir) if fn.startswith('sen') and fn.endswith('.conll')]
        self.parsed = ConllSen.__read_conll(f"{sen_dir}/parsed.conll")
        self.orig_oie_data = {
            fn.split("/")[-1].split(".json")[0].split("sen")[-1]: ConllSen.__read_conll(fn) for fn in orig_fns
        }

    @staticmethod
    def __read_conll(fn):
        with open(fn) as f:
            lines = f.readlines()
        return [line.strip().split("\t") for line in lines if line.strip()]

    def sen_text(self):
        return " ".join([line[1] for line in self.parsed])

    def len(self):
        return len(self.parsed)

    def pos_tags(self):
        return [line[3] for line in self.parsed]
