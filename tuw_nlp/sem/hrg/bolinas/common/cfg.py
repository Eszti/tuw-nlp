import itertools
import heapq


class NonterminalLabel(object):
    """
    There can be multiple nonterminal edges with the same symbol. Wrap the
    edge into an object so two edges do not compare equal.
    Nonterminal edges carry a nonterminal symbol and an index that identifies
    it uniquely in a rule.
    """

    def __init__(self, label, index=None):
        self.label = label
        self.index = index

    def __eq__(self, other):
        try:
            return self.label == other.label and self.index == other.index
        except AttributeError:
            return False

    def __repr__(self):
        return "NT(%s)" % str(self)

    def __str__(self):
        if self.index is not None:
            return "%s$%s" % (str(self.label), str(self.index))
        else:
            return "%s$" % str(self.label)

    def __hash__(self):
        return 83 * hash(self.label) + 17 * hash(self.index)

    @classmethod
    def from_string(cls, s):
        label, index = s.split("$")
        return NonterminalLabel(label, index or None)


class Chart(dict):
    """
    A CKY style parse chart that can return k-best derivations and can return inside and outside probabilities.
    """

    def derivations(self, item):
        """
        Return all derivations from this chart.
        """

        if item == "START":
            rprob = 0.0
        else:
            rprob = item.rule.weight

        # If item is a leaf, just return it and its probability
        if not item in self:
            if item == "START":
                print("No derivations.")
                return []
            else:
                return [(rprob, item)]

        pool = []
        for split in self[item]:
            nts, children = zip(*split.items())
            kbest_each_child = [self.derivations(child) for child in children]

            all_combinations = list(itertools.product(*kbest_each_child))

            combinations_for_sorting = []
            for combination in all_combinations:
                weights, trees = zip(*combination)
                try:
                    heapq.heappush(combinations_for_sorting, (sum(weights) + rprob, trees))
                except TypeError:
                    pass

            for prob, trees in combinations_for_sorting:
                new_tree = (item, dict(zip(nts, trees)))
                try:
                    heapq.heappush(pool, (prob, new_tree))
                except TypeError:
                    pass

        return sorted(pool, key=lambda x: x[0], reverse=True)
