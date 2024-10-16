# Some general advice for reading this file:
#
# Every rule specifies some fragment of the object (graph, string or both) that
# is being parsed, as well as a visit order on the individual elements of that
# fragment (tokens or edges respectively). The number of elements already
# visited is called the "size" of this item, and an item with nothing left to
# visit is "closed". The visit order specifies an implicit binarization of the
# rule in question, by allowing the item to consume only one other object (which
# we call the "outside" of the item) at any given time.
#
# In consuming this object, we either "shift" a terminal element or "complete" a
# nonterminal (actually a closed chart item). Each of these steps produces a new
# chart item.
from tuw_nlp.sem.hrg.bolinas.common.cfg import NonterminalLabel


class Item(object):
    pass


class HergItem(Item):
    """
    Chart item for a HRG parse.
    """

    def __init__(self, rule, size=None, shifted=None, mapping=None, nodeset=None, nodelabels=False):
        # by default start empty, with no part of the graph consumed
        if size == None:
            size = 0
        if shifted == None:
            shifted = frozenset()
        if mapping == None:
            mapping = dict()
        if nodeset == None:
            nodeset = frozenset()

        self.rule = rule
        self.size = size
        self.shifted = shifted
        self.mapping = mapping
        self.nodeset = nodeset

        self.rev_mapping = dict((val, key) for key, val in mapping.items())

        self.nodelabels = nodelabels

        # Store the nonterminal symbol and index of the previous complete
        # on this item so we can rebuild the derivation easily
        triples = rule.rhs1.triples(nodelabels=nodelabels)
        self.outside_symbol = None
        if size < len(triples):
            # this item is not closed
            self.outside_triple = triples[rule.rhs1_visit_order[size]]
            self.outside_edge = self.outside_triple[1]
            self.closed = False
            self.outside_is_nonterminal = isinstance(self.outside_triple[1], NonterminalLabel)
            if self.outside_is_nonterminal:
                self.outside_symbol = self.outside_triple[1].label
                self.outside_nt_index = self.outside_triple[1].index
        else:
            # this item is closed
            self.outside_triple = None
            self.outside_edge = None
            self.closed = True
            self.outside_is_nonterminal = False

        self.__cached_hash = None

    def __hash__(self):
        # memoize the hash function
        if not self.__cached_hash:
            self.__cached_hash = 2 * hash(self.rule) + 3 * self.size + \
                                 5 * hash(self.shifted)
        return self.__cached_hash

    def __eq__(self, other):
        return isinstance(other, HergItem) and \
               other.rule == self.rule and \
               other.size == self.size and \
               other.shifted == self.shifted and \
               other.mapping == self.mapping

    def __repr__(self):
        return 'HergItem(%d, %d, %s, %s)' % (self.rule.rule_id, self.size, self.rule.symbol, len(self.shifted))

    def __str__(self):
        return '[%d, %d/%d, %s, {%s}]' % (self.rule.rule_id,
                                          self.size,
                                          len(self.rule.rhs1.triples()),
                                          self.outside_symbol,
                                          str([x for x in self.shifted]))

    def can_shift(self, new_edge):
        """
        Determines whether new_edge matches the outside of this item, and can be
        shifted.
        """
        # can't shift into a closed item
        if self.closed:
            return False
        # can't shift an edge that is already inside this item
        if new_edge in self.shifted:
            return False
        olabel = self.outside_triple[1]
        nlabel = new_edge[1]
        # make sure new_edge mathes the outside label
        if olabel != nlabel:
            return False
        # make sure new_edge preserves a consistent mapping between the nodes of the
        # graph and the nodes of the rule
        if self.nodelabels:
            o1, o1_label = self.outside_triple[0]
            n1, n1_label = new_edge[0]
            if o1_label != n1_label:
                return False
        else:
            o1 = self.outside_triple[0]
            n1 = new_edge[0]

        if o1 in self.mapping and self.mapping[o1] != n1:
            return False

        # If this node is not a node of this rule RHS, but of a subgraph,
        # it needs to have a mapping otherwise, we can't attach.
        if n1 in self.nodeset and n1 not in self.rev_mapping:
            return False

        if self.nodelabels:
            if self.outside_triple[2]:
                o2, o2_labels = zip(*self.outside_triple[2])
            else:
                o2, o2_labels = [], []
            if new_edge[2]:
                n2, n2_labels = zip(*new_edge[2])
            else:
                n2, n2_labels = [], []
            if o2_labels != n2_labels:
                return False
        else:
            o2 = self.outside_triple[2]
            n2 = new_edge[2]

            if len(o2) != len(n2):
                return False

        for i in range(len(o2)):
            if o2[i] in self.mapping and self.mapping[o2[i]] != n2[i]:
                return False
            # Again, need to make sure this node is part of the rule RHS, not of a  proper subgraph.
            if n2[i] in self.nodeset and n2[i] not in self.rev_mapping:
                return False

        return True

    def shift(self, new_edge):
        """
        Creates the chart item resulting from a shift of new_edge. Assumes
        can_shift returned true.
        """
        o1 = self.outside_triple[0][0] if self.nodelabels else self.outside_triple[0]
        o2 = tuple(x[0] for x in self.outside_triple[2]) if self.nodelabels else self.outside_triple[2]

        n1 = new_edge[0][0] if self.nodelabels else new_edge[0]
        n2 = tuple(x[0] for x in new_edge[2]) if self.nodelabels else new_edge[2]

        new_nodeset = self.nodeset | set(n2) | set([n1])

        assert len(o2) == len(n2)
        new_size = self.size + 1
        new_shifted = frozenset(self.shifted | set([new_edge]))
        new_mapping = dict(self.mapping)
        new_mapping[o1] = n1
        for i in range(len(o2)):
            new_mapping[o2[i]] = n2[i]

        return HergItem(self.rule, new_size, new_shifted, new_mapping, new_nodeset, self.nodelabels)

    def can_complete(self, new_item):
        """
        Determines whether new_item matches the outside of this item (i.e. if the
        nonterminals match and the node mappings agree).
        """
        # can't add to a closed item
        if self.closed:
            return False
        # can't shift an incomplete item
        if not new_item.closed:
            return False

        # make sure labels agree
        if not self.outside_is_nonterminal:
            return False

        # Make sure items are disjoint
        if any(edge in self.shifted for edge in new_item.shifted):
            return False

        # make sure mappings agree
        if self.nodelabels:
            o1, o1label = self.outside_triple[0]
            if self.outside_triple[2]:
                o2, o2labels = zip(*self.outside_triple[2])
            else:
                o2, o2labels = [], []
        else:
            o1 = self.outside_triple[0]
            o2 = self.outside_triple[2]

        if len(o2) != len(new_item.rule.rhs1.external_nodes):
            return False

        nroot = list(new_item.rule.rhs1.roots)[0]

        # Check root label
        if self.nodelabels and o1label != new_item.rule.rhs1.node_to_concepts[nroot]:
            return False

        if o1 in self.mapping and self.mapping[o1] != new_item.mapping[nroot]:
            return False

        for i in range(len(o2)):
            otail = o2[i]
            ntail = new_item.rule.rhs1.rev_external_nodes[i]
            # Check tail label
            if self.nodelabels and o2labels[i] != new_item.rule.rhs1.node_to_concepts[ntail]:
                return False
            if otail in self.mapping and self.mapping[otail] != new_item.mapping[ntail]:
                return False

        for node in new_item.mapping.values():
            if node in self.rev_mapping:
                onode = self.rev_mapping[node]
                if not (onode == o1 or onode in o2):
                    return False

        return True

    def complete(self, new_item):
        """
        Creates the chart item resulting from a complete of new_item. Assumes
        can_shift returned true.
        """
        o1 = self.outside_triple[0][0] if self.nodelabels else self.outside_triple[0]
        o2 = tuple(x[0] for x in self.outside_triple[2]) if self.nodelabels else self.outside_triple[2]

        new_size = self.size + 1
        new_shifted = frozenset(self.shifted | new_item.shifted)
        new_mapping = dict(self.mapping)
        new_mapping[o1] = new_item.mapping[list(new_item.rule.rhs1.roots)[0]]
        for i in range(len(o2)):
            otail = o2[i]
            ntail = new_item.rule.rhs1.rev_external_nodes[i]
            new_mapping[otail] = new_item.mapping[ntail]
        new_nodeset = self.nodeset | new_item.nodeset

        new = HergItem(self.rule, new_size, new_shifted, new_mapping, new_nodeset, self.nodelabels)
        return new
