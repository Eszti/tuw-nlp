import heapq
import itertools
import time


class Chart(dict):
    """
    A CKY style parse chart that can return k-best derivations and can return inside and outside probabilities.
    """

    def derivations(self, item="START", only_first=False, max_steps=None, k_best=None):
        start_time = time.time()
        if only_first:
            derivation, steps = self._first_derivation(item)
            derivations = [derivation]
        else:
            derivations, steps = self._derivations(item, 0, max_steps, k_best)
        elapsed_time = round(time.time() - start_time, 2)
        search_summary = f"Search: {elapsed_time} sec, {steps} steps"
        print(search_summary)
        return derivations, f"{search_summary}\n"

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

    def _first_derivation(self, item):
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
        children_derivations = [self._first_derivation(child) for child in children]
        one_from_each_child, steps = zip(*children_derivations)

        weights, trees = zip(*one_from_each_child)
        new_tree = (item, dict(zip(nts, trees)))
        new_prob = (sum(weights) + rprob)
        return (new_prob, new_tree), sum(steps) + 1

    def items_length(self):
        length = 0
        splits = self.items()
        for split in splits:
            for item_dict in split[1]:
                length += len(item_dict.values())
        return length

    def log_length(self):
        return f"Chart START items len: {len(self['START'])}\n" \
               f"Chart keys len: {len(self)}\n" \
               f"Chart items len: {self.items_length()}\n"
