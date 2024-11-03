import json
from collections import defaultdict, OrderedDict


class Triplet:
    def __init__(self, triplet_dict, label_to_nodes=True):
        if label_to_nodes:
            self.label_to_nodes = {label: sorted([int(a) for a in args]) for label, args in triplet_dict.items()}
            self.node_to_label = {n: label for label, nodes in self.label_to_nodes.items() for n in nodes}
        else:
            self.node_to_label = {int(node): label for node, label in triplet_dict.items()}
            label_to_nodes_dict = defaultdict(list)
            for node, label in self.node_to_label.items():
                label_to_nodes_dict[label].append(node)
            self.label_to_nodes = label_to_nodes_dict

    def to_file(self, fn):
        with open(fn, "w") as f:
            f.write(json.dumps(OrderedDict(sorted(self.label_to_nodes.items()))))

    @staticmethod
    def from_file(fn):
        label_to_nodes_dict = json.load(open(fn))
        return Triplet(label_to_nodes_dict)

    def get_label(self, node):
        return self.node_to_label.get(node)

    def arguments(self):
        return {k: v for k, v in self.label_to_nodes.items() if k.startswith("A")}

    def predicate(self):
        return self.label_to_nodes["P"]
