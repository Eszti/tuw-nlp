from collections import defaultdict


class WiReEx:

    def __init__(self, extraction):
        self.rel = extraction["rel"]
        self.arg1 = extraction["arg1"]
        self.arg2_plus = extraction["arg2+"]
        self.score = extraction["score"]
        self.extractor = extraction["extractor"]

    def __eq__(self, other):
        return self.rel == other.rel and self.arg1 == other.arg1 and self.arg2_plus == other.arg2_plus

    def __hash__(self):
        a2 = sum([hash((a, i)) for i, a in enumerate(self.arg2_plus)])
        return hash((self.rel, self.arg1, a2))

    def to_json(self):
        return {
            "arg1": self.arg1,
            "rel": self.rel,
            "arg2+": self.arg2_plus,
            "score": self.score,
            "extractor": self.extractor,
        }


def wire_from_dict(labels):
    arg2_keys = sorted([k for k in labels.keys() if not (k == "P" or k == "O" or k == "A0")])
    return {
        "arg1": " ".join(labels["A0"]),
        "rel": " ".join(labels["P"]),
        "arg2+": [" ".join(labels[key]) for key in arg2_keys],
    }


def get_wire_extraction(extracted_labels, sen):
    words = sen.split(" ")
    labels = defaultdict(list)
    for i, word in enumerate(words):
        word_id = i + 1
        if extracted_labels[str(word_id)] != "O":
            labels[extracted_labels[str(word_id)]].append(word)
    ret = wire_from_dict(labels)
    ret["score"] = "1.0"
    ret["extractor"] = "PoC"
    return ret


def get_wire_extraction_from_conll(sen):
    labels = defaultdict(list)
    for i, fields in enumerate(sen):
        word = fields[1]
        label = fields[-1].split("-")[0]
        if label != "O":
            labels[label].append(word)
    return wire_from_dict(labels)
