from collections import defaultdict


class WiReEx(dict):

    def __init__(self, extraction):
        dict.__init__(self,
                      arg1=extraction["arg1"],
                      rel=extraction["rel"])
        self["arg2+"] = extraction["arg2+"]
        if "score" in extraction:
            self["score"] = extraction["score"]
        if "k" in extraction:
            self["k"] = extraction["k"]
        if "sen_id" in extraction:
            self["sen_id"] = extraction["sen_id"]
        if "extractor" in extraction:
            self["extractor"] = extraction["extractor"]

    def __hash__(self):
        return hash(
            (self["rel"]["text"],
             "".join(str(self["rel"]["indexes"])),
             self["arg1"]["text"],
             "".join(str(self["arg1"]["indexes"])),
             sum([hash((" ".join(a["text"]), i)) for i, a in enumerate(self["arg2+"])]),
             sum([hash(("".join(str(a["indexes"])), i)) for i, a in enumerate(self["arg2+"])])
             ))


def create_wire_arg(arg_list):
    return {
        "text": " ".join([a[0] for a in arg_list]),
        "indexes": [a[1] for a in arg_list],
    }


def wire_from_dict(labels, sen_id):
    arg2_keys = sorted([k for k in labels.keys() if not (k == "P" or k == "O" or k == "A0")])
    return WiReEx({
        "arg1": create_wire_arg(labels["A0"]),
        "rel": create_wire_arg(labels["P"]),
        "arg2+": [create_wire_arg(labels[key]) for key in arg2_keys],
        "sen_id": sen_id,
    })


def get_wire_extraction(extracted_labels, sen_txt, k, sen_id, score="1.0", extractor="PoC"):
    words = sen_txt.split(" ")
    labels = defaultdict(list)
    for i, word in enumerate(words):
        word_id = i + 1
        if extracted_labels[str(word_id)] != "O":
            labels[extracted_labels[str(word_id)]].append((word, word_id))
    ret = wire_from_dict(labels, sen_id)
    ret["score"] = score
    ret["k"] = k
    ret["extractor"] = extractor
    return ret


def wire_from_conll(sen, sen_id):
    labels = defaultdict(list)
    for i, fields in enumerate(sen):
        word = fields[1]
        label = fields[-1].split("-")[0]
        idx = int(fields[0]) + 1
        if label != "O":
            labels[label].append((word, idx))
    return wire_from_dict(labels, sen_id)
