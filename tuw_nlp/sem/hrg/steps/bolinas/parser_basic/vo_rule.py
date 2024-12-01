from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.common.nonterminal import NonterminalLabel
from tuw_nlp.sem.hrg.steps.bolinas.common.rule import Rule


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

    def __repr__(self):
        return 'VoRule(%d,%s)' % (self.rule_id, self.symbol)

    def __hash__(self):
        return self.rule_id

    def __eq__(self, other):
        return isinstance(other, VoRule) and self.rule_id == other.rule_id
