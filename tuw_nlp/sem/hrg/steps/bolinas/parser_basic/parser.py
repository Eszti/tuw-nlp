import time
from collections import defaultdict, deque

from ordered_set import OrderedSet

from tuw_nlp.sem.hrg.steps.bolinas.common.chart import Chart
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_item import HergItem


class Parser:
    """
    A deductive style parser for hypergraphs and strings that matches parts
    of the input hypergraph according to an arbitrary visit order for edges.
    (or left-to-right for strings, in which case this is essentially
    a CKY parser).
    """

    def __init__(self, grammar, stop_at_first=False, max_steps=None, permutations=False):
        self.grammar = grammar
        self.nodelabels = grammar.nodelabels
        self.max_steps = max_steps
        self.stop_at_first = stop_at_first
        self.permutations = permutations

    def parse_graphs(self, graph_iterator, partial=False):
        """
        Parse all the graphs in graph_iterator.
        This is a generator.
        """
        for graph in graph_iterator:
            log = ""
            raw_chart, parse_log = self.parse(graph, partial=partial)
            log += parse_log
            chart, cky_log = get_cky_chart(raw_chart, self.permutations)
            log += cky_log
            log += chart.log_length()
            yield chart, log

    def parse(self, graph, partial=False):
        """
        Parses the given string and/or graph.
        """

        # This is a long function, so let's start with a high-level overview. This is
        # a "deductive-proof-style" parser: We begin with one "axiomatic" chart item
        # for each rule, and combine these items with each other and with fragments of
        # the object(s) being parsed to deduce new items. We can think of these items
        # as defining a search space in which we need to find a path to the goal item.
        # The parser implemented here performs a BFS of this search space.

        grammar = self.grammar

        parse_log = ""
        start_time = time.time()
        graph_size = len(graph.triples(nodelabels=self.nodelabels))

        # initialize data structures and lookups
        # we use various tables to provide constant-time lookup of fragments available
        # for shifting, completion, etc.
        chart = defaultdict(set)

        pgrammar = [grammar[r] for r in grammar.reachable_rules(graph, None)]

        queue = deque()
        pending = set()
        attempted = set()
        visited = set()
        nonterminal_lookup = defaultdict(OrderedSet)
        reverse_lookup = defaultdict(OrderedSet)
        edge_terminal_lookup = defaultdict(set)
        for edge in graph.triples(nodelabels=self.nodelabels):
            edge_terminal_lookup[edge[1]].add(edge)

        for rule in pgrammar:
            axiom = HergItem(rule, nodelabels=self.nodelabels)
            queue.append(axiom)
            pending.add(axiom)
            if axiom.outside_is_nonterminal:
                reverse_lookup[axiom.outside_symbol].add(axiom)

        max_queue_size = 0
        max_queue_diff_comp = 0
        max_queue_diff_outside_nt = 0
        max_queue_diff_shift = 0
        steps = 0

        # parse
        while queue:
            if self.max_steps and steps >= self.max_steps:
                break

            steps += 1

            if len(queue) > max_queue_size:
                max_queue_size = len(queue)

            item = queue.popleft()
            pending.remove(item)
            visited.add(item)

            if item.closed:
                # check if it's a complete derivation
                if self.successful_parse(item, graph_size):
                    chart['START'].add((item,))
                    if self.stop_at_first:
                        break
                elif partial and self.grammar.start_symbol == item.rule.symbol:
                    chart['START'].add((item,))

                # add to nonterminal lookup
                nonterminal_lookup[item.rule.symbol].add(item)

                # wake up any containing rules
                # Unlike in ordinary state-space search, it's possible that we will have
                # to re-visit items which couldn't be merged with anything the first time
                # we saw them, and are waiting for the current item. The reverse_lookup
                # indexes all items by their outside symbol, so we re-append to the queue
                # all items looking for something with the current item's symbol.
                before = len(queue)
                for ritem in reverse_lookup[item.rule.symbol]:
                    if ritem not in pending:
                        queue.append(ritem)
                        pending.add(ritem)
                after = len(queue)
                if (after - before) > max_queue_diff_comp:
                    max_queue_diff_comp = after - before

            else:
                if item.outside_is_nonterminal:
                    # complete
                    reverse_lookup[item.outside_symbol].add(item)

                    before = len(queue)
                    for oitem in nonterminal_lookup[item.outside_symbol]:
                        if (item, oitem) in attempted:
                            # don't repeat combinations we've tried before
                            continue
                        attempted.add((item, oitem))
                        if not item.can_complete(oitem):
                            continue
                        nitem = item.complete(oitem)
                        chart[nitem].add((item, oitem))
                        if nitem not in pending and nitem not in visited:
                            queue.append(nitem)
                            pending.add(nitem)
                    after = len(queue)
                    if (after - before) > max_queue_diff_outside_nt:
                        max_queue_diff_outside_nt = after - before

                else:
                    # shift
                    assert graph
                    new_items = [item.shift(edge) for edge in
                                 edge_terminal_lookup[item.outside_edge] if
                                 item.can_shift(edge)]

                    before = len(queue)
                    for nitem in new_items:
                        chart[nitem].add((item,))
                        if nitem not in pending and nitem not in visited:
                            queue.append(nitem)
                            pending.add(nitem)
                    after = len(queue)
                    if (after - before) > max_queue_diff_shift:
                        max_queue_diff_shift = after - before

        elapsed_time = round(time.time() - start_time, 2)
        print(f"Elapsed time for parsing: {elapsed_time} sec")
        parse_log += f"Elapsed time for parsing: {elapsed_time} sec\n"
        parse_log += f"Max queue size: {max_queue_size}\n"
        parse_log += f"Max queue diff comp: {max_queue_diff_comp}\n"
        parse_log += f"Max queue diff outside nt: {max_queue_diff_outside_nt}\n"
        parse_log += f"Max queue diff shift: {max_queue_diff_shift}\n"
        parse_log += f"Steps: {steps}\n"

        return chart, parse_log

    def successful_parse(self, item, graph_size):
        """
        Determines whether the given item represents a complete derivation of the
        object(s) being parsed.
        """
        if self.grammar.start_symbol != item.rule.symbol:
            return False
        return len(item.shifted) == graph_size


def get_cky_chart(chart, permutations):
    """
    Convert the chart returned by the parser into a standard parse chart.
    """
    cky_log = ""
    start_time = time.time()

    def search_productions(citem, chart):
        if len(chart[citem]) == 0:
            return []
        if citem == "START":
            return [{"START": child[0]} for child in chart[citem]]

        prodlist = list(chart[citem])

        lengths = set(len(x) for x in prodlist)
        assert len(lengths) == 1
        split_len = lengths.pop()

        # figure out all items that could have been used to complete this nonterminal
        if split_len != 1:
            assert split_len == 2
            symbol = prodlist[0][0].outside_symbol, prodlist[0][0].outside_nt_index
            result = []
            for child in prodlist:
                other_nts = search_productions(child[0], chart)
                if other_nts:
                    for option in other_nts:
                        d = dict(option)
                        d[symbol] = child[1]
                        result.append(d)
                else:
                    result.append(dict([(symbol, child[1])]))
            return result
        else:
            return search_productions(prodlist[0][0], chart)

    stack = ['START']
    visit_items = set()
    while stack:
        item = stack.pop()
        if item in visit_items:
            continue
        visit_items.add(item)
        for production in chart[item]:
            for citem in production:
                stack.append(citem)

    cky_chart = Chart()
    for item in visit_items:
        if not (item == 'START' or item.closed):
            continue

        prods = search_productions(item, chart)
        if prods:
            if permutations:
                cky_chart[item] = prods
            else:
                unique_prods = filter_permutations(prods)
                cky_chart[item] = unique_prods
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time for cky convertion: {elapsed_time} sec")
    cky_log += f"Elapsed time for cky convertion: {elapsed_time} sec\n"
    return cky_chart, cky_log


def filter_permutations(prods):
    unique_prods = []
    seen_combinations = set()
    for prod in prods:
        rules = sorted([f"r{item.rule.rule_id}" for item in prod.values()])
        nodes = []
        for item in prod.values():
            nodes += set(item.nodeset)
        nodes = sorted(set(nodes))
        combination = "_".join(rules + nodes)
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            unique_prods.append(prod)
    return unique_prods
