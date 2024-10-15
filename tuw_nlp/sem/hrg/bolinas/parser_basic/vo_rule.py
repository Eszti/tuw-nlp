from tuw_nlp.sem.hrg.bolinas.common.cfg import NonterminalLabel
from tuw_nlp.sem.hrg.bolinas.common.exceptions import BinarizationException
from tuw_nlp.sem.hrg.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.bolinas.common.rule import Rule


class VoRule(Rule):
    """
    A rule that stores a simple visit order for the graph.
    """

    def __init__(
            self,
            rule_id,
            symbol,
            weight,
            rhs1,
            rhs2,
            rhs1_visit_order=None,
            rhs2_visit_order=None,
            original_index=None,
            nodelabels=False,
            logprob=False
    ):
        self.rule_id = rule_id
        self.symbol = symbol
        self.weight = weight
        self.rhs1 = rhs1
        self.rhs2 = rhs2
        self.nodelabels = nodelabels
        self.logprob = logprob

        if isinstance(rhs1, list):
            self.string = rhs1

            # Set default visit order: canonical order of hyperedges or string tokens left-to-right
        # Also determine if this RHS is a terminal
        if isinstance(rhs1, Hgraph):
            assert len(rhs1.roots) == 1
            self.is_terminal = not any(rhs1.nonterminal_edges())
            self.rhs1_visit_order = rhs1_visit_order if rhs1_visit_order is not None else range(
                len(rhs1.triples(nodelabels=nodelabels)))
        else:
            self.is_terminal = not any([t for t in rhs1 if type(t) is NonterminalLabel])
            self.rhs1_visit_order = rhs1_visit_order if rhs1_visit_order is not None else range(len(rhs1))

        if self.rhs2 is not None:
            if isinstance(rhs2, Hgraph):
                self.rhs2_visit_order = rhs2_visit_order if rhs2_visit_order is not None else range(
                    len(rhs2.triples(nodelabels=nodelabels)))
            else:
                self.rhs2_visit_order = rhs2_visit_order if rhs2_visit_order is not None else range(len(rhs2))

        if original_index != None:
            self.original_index = original_index
        else:
            self.original_index = None

    def project_right(self):
        return VoRule(self.rule_id, self.symbol, self.weight, self.rhs2, None, \
                      rhs1_visit_order=self.rhs2_visit_order, rhs2_visit_order=None, \
                      original_index=self.original_index, nodelabels=self.nodelabels)

    def project_left(self):
        return VoRule(self.rule_id, self.symbol, self.weight, self.rhs1, None, \
                      rhs1_visit_order=self.rhs1_visit_order, rhs2_visit_order=None, \
                      original_index=self.original_index, nodelabels=self.nodelabels)

    def reweight(self, nweight):
        return VoRule(self.rule_id, self.symbol, nweight, self.rhs1, self.parse, \
                      self.rhs1_visit_order, self.rhs2_visit_order)

    def canonicalize_amr(self):
        return VoRule(self.rule_id, self.symbol, self.weight,
                      self.amr.clone_canonical(), self.parse, self.rhs1_visit_order,
                      self.rhs2_visit_order)

    def __repr__(self):
        return 'VoRule(%d,%s)' % (self.rule_id, self.symbol)

    def __hash__(self):
        return self.rule_id

    def __eq__(self, other):
        return isinstance(other, VoRule) and self.rule_id == other.rule_id

    def binarize(self, next_id):
        oid = next_id
        tree = self.parse
        amr = self.amr

        # handle all-terminal rules
        if not any(s[0] == '#' for s in tree.leaves()):
            return [VoRule(next_id, self.symbol, self.weight, self.amr, self.parse,
                           self.rhs1_visit_order, self.rhs2_visit_order)], next_id + 1

        # handle rules containing nonterminals
        rules = []
        try:
            tree, amr, at_rules, next_id = self.collapse_amr_terminals(tree, amr,
                                                                       next_id)
            rules += at_rules

            string = tree.leaves()

            string, amr, st_rules, next_id = self.collapse_string_terminals(string,
                                                                            amr, next_id)
            rules += st_rules

            string, amr, nt_rules, next_id = self.merge_string_nonterminals(string,
                                                                            amr, next_id)
            rules += nt_rules
        except BinarizationException:
            print('Unbinarizable rule!')
            return None, oid

        # sanity check---did we completely binarize the rule?
        assert len(string) == 1
        assert len(amr.triples()) == 1
        rules.append(VoRule(next_id + 1, self.symbol, self.weight, amr, string[0]))
        return rules, next_id + 2

    def binarize_tree(self, next_id):
        oid = next_id
        tree = self.parse
        amr = self.amr

        # handle all-terminal rules
        if not any(s[0] == '#' for s in tree.leaves()):
            return [VoRule(next_id, self.symbol, self.weight, self.amr, self.parse,
                           self.rhs1_visit_order, self.rhs2_visit_order)], next_id + 1

        # handle rules containing nonterminals
        rules = []
        try:
            tree, amr, at_rules, next_id = self.collapse_amr_terminals(tree, amr,
                                                                       next_id)
            rules += at_rules

            tree, amr, ts_rules, next_id = self.merge_tree_symbols(tree, amr, next_id)
            rules += ts_rules
        except BinarizationException:
            print('Unbinarizable rule!')
            return None, oid

        # sanity check as above
        assert isinstance(tree, str)
        assert len(amr.triples()) == 1
        rules.append(VoRule(next_id + 1, self.symbol, self.weight, amr, tree))
        return rules, next_id + 2

    def terminal_search(self, root, triples):
        """
    Searches for terminal edges reachable from the given root edge without
    passing through a nonterminal edge.
    """
        stack = []
        for r in root[2]:
            stack.append(r)
        out = set()
        while stack:
            top = stack.pop()
            children = [t for t in triples if t[0] == top and not isinstance(t[1],
                                                                             NonterminalLabel) and t not in out]
            for c in children:
                out.add(c)
                for t in c[2]:
                    stack.append(t)
        return out
