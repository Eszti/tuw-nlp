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
