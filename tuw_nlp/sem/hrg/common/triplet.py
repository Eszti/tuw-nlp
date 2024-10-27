import json
from collections import OrderedDict


class Triplet:
    def __init__(self, predicate, arguments):
        self.predicate = predicate
        self.arguments = arguments
        self.node_to_label = {p: "P" for p in self.predicate}
        for l, nodes in self.arguments.items():
            for n in nodes:
                assert n not in self.node_to_label
                self.node_to_label[n] = l

    def to_file(self, fn):
        with open(fn, "w") as f:
            f.write(f"{json.dumps(self.predicate)}\n")
            f.write(f"{json.dumps(OrderedDict(sorted(self.arguments.items())))}\n")

    @staticmethod
    def from_file(fn):
        with open(fn) as f:
            lines = f.readlines()
        assert len(lines) == 2
        predicate = json.loads(lines[0])
        arguments = json.loads(lines[1])
        return Triplet(predicate, arguments)

    def get_label(self, node):
        return self.node_to_label[node] if node in self.node_to_label else None
