import itertools
import heapq
import time
from datetime import datetime

from tuw_nlp.sem.hrg.common.io import log_to_console_and_log_lines


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

    def derivations(self, log_lines, item="START", max_steps=None, k_best=None):
        start_time = time.time()
        log_lines.append(f"Start search: {datetime.now()}\n")
        derivations, steps = self._derivations(item, 0, max_steps, k_best)
        log_lines.append(f"Finish search: {datetime.now()}\n")
        etime = time.time() - start_time
        log_to_console_and_log_lines(f"Elapsed time for searching: {round(etime, 2)} sec", log_lines)
        log_to_console_and_log_lines(f"Used steps: {steps}", log_lines)
        return derivations

    def _derivations(self, item, done_steps, max_steps, k_best):
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
                return [], 1
            else:
                return [(rprob, item)], 1

        pool = []
        all_steps = done_steps
        splits = self[item]
        for split in splits:
            if max_steps is None or all_steps < max_steps:
                nts, children = zip(*split.items())
                children_derivations = [self._derivations(child, all_steps, max_steps, k_best) for child in children]
                kbest_each_child, steps_each_child = zip(*children_derivations)
                all_steps += sum(steps_each_child)

                all_combinations = list(itertools.product(*kbest_each_child))

                combinations_for_sorting = []
                for combination in all_combinations:
                    weights, trees = zip(*combination)
                    try:
                        heapq.heappush(combinations_for_sorting, (sum(weights) + rprob, trees))
                    except TypeError:
                        pass

                    for prob, trees in sorted(combinations_for_sorting, key=lambda x: x[0], reverse=True)[:k_best]:
                        new_tree = (item, dict(zip(nts, trees)))
                        try:
                            heapq.heappush(pool, (prob, new_tree))
                        except TypeError:
                            pass

        return sorted(pool, key=lambda x: x[0], reverse=True)[:k_best], all_steps - done_steps + 1

    def first_derivation(self, item):
        """
        Return one derivation from this chart.
        """
        if item == "START":
            rprob = 0.0
        else:
            rprob = item.rule.weight

        if not item in self:
            if item == "START":
                print("No derivations.")
                return None
            else:
                return (rprob, item), 1

        split = list(self[item])[0]
        nts, children = zip(*split.items())
        children_derivations = [self.first_derivation(child) for child in children]
        one_from_each_child, steps = zip(*children_derivations)

        weights, trees = zip(*one_from_each_child)
        new_tree = (item, dict(zip(nts, trees)))
        new_prob = (sum(weights) + rprob)
        return (new_prob, new_tree), sum(steps)+1

    def items_length(self):
        length = 0
        splits = self.items()
        for split in splits:
            for item_dict in split[1]:
                length += len(item_dict.values())
        return length

