from collections import defaultdict, OrderedDict
import re
import sys

from tuw_nlp.sem.hrg.steps.bolinas.common.cfg import NonterminalLabel


class ListMap(defaultdict):
    """
    A  map that can contain several values for the same key.
    @author: Nathan Schneider (nschneid)
    @since: 2012-06-18

    >>> x = ListMap()
    >>> x.append('mykey', 3)
    >>> x.append('key2', 'val')
    >>> x.append('mykey', 8)
    >>> x
    defaultdict(<type 'list'>, {'key2': ['val'], 'mykey': [3, 8]})
    >>> x['mykey']
    3
    >>> x.getall('mykey')
    [3, 8]
    >>> x.items()
    [('key2', 'val'), ('mykey', 3), ('mykey', 8)]
    >>> x.itemsfor('mykey')
    [('mykey', 3), ('mykey', 8)]
    >>> x.replace('mykey', 0)
    >>> x
    defaultdict(<type 'list'>, {'key2': ['val'], 'mykey': [0]})
    """

    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, list, *args, **kwargs)

    def __setitem__(self, k, v):
        if k in self:
            raise KeyError('Cannot assign to ListMap entry; use replace() or append()')
        return defaultdict.__setitem__(self, k, v)

    def __getitem__(self, k):
        """Returns the *first* list entry for the key."""
        return dict.__getitem__(self, k)[0]

    def getall(self, k):
        return dict.__getitem__(self, k)

    def items(self):
        return [(k, v) for k, vv in defaultdict.items(self) for v in vv]

    def values(self):
        return [v for k, v in self.items()]

    def itemsfor(self, k):
        return [(k, v) for v in self.getall(k)]

    def replace(self, k, v):
        defaultdict.__setitem__(self, k, [v])

    def append(self, k, v):
        defaultdict.__getitem__(self, k).append(v)

    def remove(self, k, v):
        defaultdict.__getitem__(self, k).remove(v)
        if not dict.__getitem__(self, k):
            del self[k]

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + t[2:]


# Actual AMR class

class Hgraph(defaultdict):
    """
    An abstract meaning representation.
    The structure consists of nested mappings from role names to fillers.
    Because a concept may have multiple roles with the same name,
    a ListMap data structure holds a list of fillers for each role.
    A set of (concept, role, filler) triples can be extracted as well.
    """
    _parser_singleton = None

    def __init__(self, *args, **kwargs):

        defaultdict.__init__(self, ListMap, *args, **kwargs)
        self.roots = []
        self.external_nodes = {}
        self.rev_external_nodes = {}

        # Count how many replacements have occurred in this DAG
        # to prefix unique new node IDs for glued fragments.
        self.replace_count = 0

        self.__cached_triples = None
        self.__cached_depth = None
        self.__nodelabels = False

        self.node_alignments = {}
        self.edge_alignments = {}

        self.__cached_triples = None
        self.node_to_concepts = {}

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + (self.__dict__,) + t[3:]

    def _get_node_hashes(self):
        tabu = set()
        queue = []
        node_to_id = defaultdict(int)
        for x in sorted(self.roots):
            if type(x) is tuple:
                for y in x:
                    queue.append((y, 0))
                    node_to_id[y] = 0
            else:
                queue.append((x, 0))
                node_to_id[x] = 0
        while queue:
            node, depth = queue.pop(0)
            if node not in tabu:
                tabu.add(node)
                rels = tuple(sorted(self[node].keys()))
                node_to_id[node] += 13 * depth + hash(rels)

                for rel in rels:
                    children = self[node].getall(rel)
                    for child in children:
                        if child not in node_to_id:
                            if type(child) is tuple:
                                for c in child:
                                    queue.append((c, depth + 1))
                            else:
                                queue.append((child, depth + 1))
        return node_to_id

    def __hash__(self):
        # We compute a hash for each node in the AMR and then sum up the hashes.
        # Collisions are minimized because each node hash is offset according to its distance from
        # the root.
        node_to_id = self._get_node_hashes()
        return sum(node_to_id[node] for node in node_to_id)

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def from_string(cls, amr_string):
        """
        Initialize a new abstract meaning representation from a Pennman style string.
        """
        if not cls._parser_singleton:  # Initialize the AMR parser only once
            from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.graph_description_parser import GraphDescriptionParser
            _parser_singleton = GraphDescriptionParser()
            amr = _parser_singleton.parse_string(amr_string)
            return amr

    @classmethod
    def from_triples(cls, triples, concepts, roots=None, warn=sys.stderr):
        """
        Initialize a new hypergraph from a collection of triples and a node to concept map.
        """

        graph = Hgraph()  # Make new DAG

        for parent, relation, child in triples:
            if isinstance(parent, str):
                new_par = parent.replace("@", "")
                if parent.startswith("@"):
                    graph.external_nodes.append(new_par)
            else:
                new_par = parent

            if type(child) is tuple:
                new_child = []
                for c in child:
                    if isinstance(c, str):
                        new_c = c.replace("@", "")
                        new_child.append(new_c)
                        if c.startswith("@"):
                            graph.external_nodes.append(new_c)
                    else:
                        new_child.append(c)
                new_child = tuple(new_child)
            else:
                # Allow triples to have single string children for convenience.
                # and downward compatibility.
                if isinstance(child, str):
                    tmpchild = child.replace("@", "")
                    if child.startswith("@"):
                        graph.external_nodes.append(tmpchild)
                    new_child = (tmpchild,)
                else:
                    new_child = (child,)

            graph._add_triple(new_par, relation, new_child, warn=warn)

        # Allow the passed root to be either an iterable of roots or a single root
        if roots:
            try:  # Try to interpret roots as iterable
                graph.roots.extend(roots)
            except TypeError:  # If this fails just use the whole object as root
                graph.roots = list([roots])
        else:
            graph.roots = graph.find_roots(warn=warn)

        graph.node_to_concepts = concepts
        graph.__cached_triples = None
        return graph

    def get_nodes(self):
        """
        Return the set of node identifiers in the DAG.
        """
        ordered = self.get_ordered_nodes()
        res = OrderedDict(sorted(ordered.items(), key=lambda x: x[1]))
        return res

    def nonterminal_edges(self):
        """
        Retrieve all nonterminal labels from the DAG.
        """
        return [t for t in self.triples() if isinstance(t[1], NonterminalLabel)]

    def get_terminals_and_nonterminals(self, nodelabels=False):
        """
        Return a tuple in which the first element is a set of all terminal labels
        and the second element is a set of all nonterminal labels.
        """
        # This is used to compute reachability of grammar rules
        terminals = set()
        nonterminals = set()
        for p, r, children in self.triples():
            if isinstance(r, NonterminalLabel):
                nonterminals.add(r.label)
            else:
                if nodelabels:
                    terminals.add((self.node_to_concepts[p], r, tuple([self.node_to_concepts[c] for c in children])))
                else:
                    terminals.add(r)
        return terminals, nonterminals

    def reach(self, node):
        """
        Return the set of nodes reachable from a node
        """
        res = set()
        for p, r, c in self.triples(start_node=node, instances=False):
            res.add(p)
            if type(c) is tuple:
                res.update(c)
            else:
                res.add(c)
        return res

    def find_roots(self, warn=sys.stderr):
        """
        Find and return a set of the roots of the DAG. This does NOT set the 'roots' attribute.
        """
        # there cannot be an odering of root nodes so it is okay to return a set
        parents = set()
        for k in self.keys():
            if type(k) is tuple:
                parents.update(k)
            else:
                parents.add(k)
        children = set()
        for node in parents:
            for v in self[node].values():
                if type(v) is tuple:
                    children.update(v)
                else:
                    children.add(v)
        roots = list(parents - children)

        not_found = parents.union(children)
        for r in roots:
            x = self.triples(start_node=r, instances=False)
            for p, r, c in x:
                if p in not_found:
                    not_found.remove(p)
                if type(c) is tuple:
                    for ch in c:
                        if ch in not_found:
                            not_found.remove(ch)
                if c in not_found:
                    not_found.remove(c)

        while not_found:
            parents = sorted([x for x in not_found if self[x]], key=lambda a: len(self.triples(start_node=a)))
            if not parents:
                if warn: warn.write("WARNING: orphaned leafs %s.\n" % str(not_found))
                roots.extend(list(not_found))
                return roots
            new_root = parents.pop()
            for p, r, c in self.triples(start_node=new_root):
                if p in not_found:
                    not_found.remove(p)
                if type(c) is tuple:
                    for ch in c:
                        if ch in not_found:
                            not_found.remove(ch)
                if c in not_found:
                    not_found.remove(c)
            roots.append(new_root)
        return roots

    def get_ordered_nodes(self):
        """
        Get an mapping of nodes in this DAG to integers specifying a total order of 
        nodes. (partial order broken according to edge_label).
        """
        order = {}
        count = 0
        for par, rel, child in self.triples(instances=False):
            if not par in order:
                order[par] = count
                count += 1
            if type(child) is tuple:
                for c in child:
                    if not c in order:
                        order[c] = count
                        count += 1
            else:
                if not child in order:
                    order[child] = count
                    count += 1
        return order

    def find_leaves(self):
        """
        Get all leaves in a DAG.
        """
        out_count = defaultdict(int)
        for par, rel, child in self.triples():
            out_count[par] += 1
            if type(child) is tuple:
                for c in child:
                    if not c in out_count:
                        out_count[c] = 0
            else:
                if not child in out_count:
                    out_count[child] = 0
        result = [n for n in out_count if out_count[n] == 0]
        order = self.get_ordered_nodes()
        result.sort(key=lambda x: order[x])
        return result

    def get_reentrant_nodes(self):
        """
        Get a list of nodes that have an in-degree > 1.
        """
        in_count = defaultdict(int)
        for par, rel, child in self.triples():
            if type(child) is tuple:
                for c in child:
                    in_count[c] += 1
            else:
                in_count[child] += 1
        result = [n for n in in_count if in_count[n] > 1]
        order = self.get_ordered_nodes()
        result.sort(key=lambda x: order[x])
        return result

    def dfs(
            self,
            extractor=lambda node, firsthit, leaf: node.__repr__(),
            combiner=lambda par, childmap, depth: {par: childmap.items()}, 
            hedge_combiner=lambda x: tuple(x)
    ):
        """
        Recursively traverse the dag depth first starting at node. When traveling up through the
        recursion a value is extracted from each child node using the provided extractor method,
        then the values are combined using the provided combiner method. At the root node the
        result of the combiner is returned. Extractor takes a "firsthit" argument that is true
        the first time a node is touched.
        """
        tabu = set()
        tabu_edge = set()

        def rec_step(node, depth):
            if type(node) is tuple:
                pass
            else:
                node = (node,)
            allnodes = []
            for n in node:
                firsthit = not n in tabu
                tabu.add(n)
                leaf = False if self[n] else True
                extracted = extractor(n, firsthit, leaf)
                child_map = ListMap()
                for rel, child in self[n].items():
                    if not (n, rel, child) in tabu_edge:
                        if child in tabu:
                            child_map.append(rel, extractor(child, False, leaf))
                            # pass
                        else:
                            tabu_edge.add((n, rel, child))
                            child_map.append(rel, rec_step(child, depth + 1))
                if child_map:
                    combined = combiner(extracted, child_map, depth)
                    allnodes.append(combined)
                else:
                    allnodes.append(extracted)
            return hedge_combiner(allnodes)

        return [rec_step(node, 0) for node in self.roots]

    def triples(self, instances=False, start_node=None, refresh=False, nodelabels=False):
        """
        Retrieve a list of (parent, edge-label, tails) triples. 
        """

        if (not (refresh or start_node or nodelabels != self.__nodelabels)) and self.__cached_triples:
            return self.__cached_triples

        triple_to_depth = {}
        triples = []
        tabu = set()

        if start_node:
            queue = [(start_node, 0)]
        else:
            queue = [(x, 0) for x in self.roots]
        while queue:
            node, depth = queue.pop(0)
            if not node in tabu:
                tabu.add(node)
                for rel, child in sorted(self[node].items(), key=lambda x: str(x[0])):
                    if nodelabels:
                        newchild = tuple([(n, self.node_to_concepts[n]) for n in child])
                        newnode = (node, self.node_to_concepts[node])
                        t = (newnode, rel, newchild)
                    else:
                        t = (node, rel, child)

                    triples.append(t)
                    triple_to_depth[t] = depth
                    if type(child) is tuple:
                        for c in child:
                            if not c in tabu:
                                queue.append((c, depth + 1))
                    else:
                        if not child in tabu:
                            queue.append((child, depth + 1))

        if not start_node:
            self.__cached_triples = triples
            self.__cached_depth = triple_to_depth
            self.__nodelabels = nodelabels

        return triples

    def __str__(self):
        return self.to_bolinas_str()

    def to_bolinas_str(self, nodeids=False):

        nodeids_to_print = self.get_reentrant_nodes()
        if nodeids:
            nodeids_to_print = self.get_nodes()

        def extractor(node, firsthit, leaf):
            if node is None:
                return "root"
            if type(node) is tuple or type(node) is list:
                return " ".join("%s*%i" % (n, self.external_nodes[n]) if n in self.external_nodes else n for n in node)
            else:
                if type(node) is int or type(node) is float or isinstance(node, (Literal, StrLiteral, Quantity)):
                    return str(node)
                else:
                    if firsthit:
                        if node in self.node_to_concepts and self.node_to_concepts[node]:
                            concept = self.node_to_concepts[node]
                            if node in self.external_nodes:
                                return "%s%s*%i " % (
                                    "%s." % node if node in nodeids_to_print else "", concept,
                                    self.external_nodes[node])
                            else:
                                return "%s%s " % ("%s." % node if node in nodeids_to_print else "", concept)
                        else:
                            if node in self.external_nodes:
                                return "%s.*%i " % (node if node in nodeids_to_print else "", self.external_nodes[node])
                            else:
                                return "%s." % (node if node in nodeids_to_print else "")
                    else:
                        return "%s." % (node if node in nodeids_to_print else "")

        def combiner(nodestr, childmap, depth):
            nt_children = sorted(
                [child for child in childmap.items() if type(child[0]) == NonterminalLabel],
                key=lambda x: str(x[0])
            )
            term_children = sorted([child for child in childmap.items() if type(child[0]) == str])
            childmap_items = nt_children + term_children
            childstr_list = ["\n%s %s %s" % (depth * "\t", ":%s" % rel if rel else "", child) for rel, child in childmap_items]
            childstr = " ".join(childstr_list)
            return "(%s %s)" % (nodestr, childstr)

        def hedgecombiner(nodes):
            return " ".join(nodes)

        return " ".join(self.dfs(extractor, combiner, hedgecombiner))

    def to_string(self, newline=False):
        if newline:
            return str(self)
        else:
            return re.sub(r"(\n|\s+)", " ", str(self))

    def _add_triple(self, parent, relation, child, warn=sys.stderr):
        """
        Add a (parent, relation, child) triple to the DAG.
        """
        if type(child) is not tuple:
            child = (child,)
        if parent in child:
            if warn: warn.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
        for c in child:
            x = self[c]
            for rel, test in self[c].items():
                if parent in test:
                    if warn: warn.write("WARNING: (%s, %s, %s) produces a cycle with (%s, %s, %s)\n" % (
                        parent, relation, child, c, rel, test))
        self[parent].append(relation, child)


class StrLiteral(str):
    def __str__(self):
        return '"%s"' % "".join(self)

    def __repr__(self):
        return "".join(self)


class Quantity(str):
    pass


class Literal(str):
    def __str__(self):
        return "'%s" % "".join(self)

    def __repr__(self):
        return "".join(self)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
