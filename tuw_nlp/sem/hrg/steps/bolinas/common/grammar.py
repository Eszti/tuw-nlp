from collections import defaultdict
import math
from io import StringIO

from tuw_nlp.sem.hrg.steps.bolinas.common.cfg import NonterminalLabel
from tuw_nlp.sem.hrg.steps.bolinas.common.exceptions import GrammarError, ParserError
from tuw_nlp.sem.hrg.steps.bolinas.common.hgraph.hgraph import Hgraph
from tuw_nlp.sem.hrg.steps.bolinas.parser_basic.vo_rule import VoRule

GRAPH_FORMAT = "hypergraph"
STRING_FORMAT = "string"
TREE_FORMAT = "tree"


def parse_string(s):
    """
    Parse the RHS of a CFG rule.
    """
    tokens = s.strip().split()
    res = []
    nt_index = 0
    for t in tokens:
        if "$" in t:
            new_token = NonterminalLabel.from_string(t)
            if not new_token.index:
                new_token.index = "_%i" % nt_index
                nt_index = nt_index + 1
        else:
            new_token = t
        res.append(new_token)
    return res


def _terminals_and_nts_from_string(string):
    terminals = set()
    nonterminals = set()
    for tok in string:
        if isinstance(tok, NonterminalLabel):
            nonterminals.add(tok.label)
        else:
            terminals.add(tok)
    return terminals, nonterminals


class Grammar(dict):
    """
    Represents a set of rules as a mapping from rule IDs to rules and defines
    operations to be performed on the entire grammar.
    """

    def __init__(self, nodelabels=False, logprob=False):
        super().__init__()
        self.nodelabels = nodelabels
        self.start_symbol = "truth"
        self.logprob = logprob

        self.lhs_to_rules = defaultdict(set)
        self.nonterminal_to_rules = defaultdict(set)
        self.rhs1_terminal_to_rules = defaultdict(set)
        self.rhs2_terminal_to_rules = defaultdict(set)
        self.startsymbol = None

    @classmethod
    def load_from_file(cls, in_file, rule_class=VoRule, reverse=False, nodelabels=False, logprob=False):
        """
        Loads a SHRG grammar from the given file.
        See documentation for format details.

        rule_class specifies the type of rule to use. VoRule is a subclass using an arbitrary graph
        visit order (also used for strings). TdRule computes a tree decomposition on the first RHS
        when initialized.
        """

        output = Grammar(nodelabels=nodelabels, logprob=logprob)

        rule_count = 1
        line_count = 0
        is_synchronous = False

        rhs1_type = None
        rhs2_type = None

        buf = StringIO()

        for line in in_file:
            line_count += 1
            l = line.strip()
            if l:
                if "#" in l:
                    content, comment = l.split("#", 1)
                else:
                    content = l
                buf.write(content.strip())
                if ";" in content:
                    rulestring = buf.getvalue()
                    try:
                        content, weights = rulestring.split(";", 1)
                        weight = 0.0 if not weights else (float(weights) if logprob else math.log(float(weights)))
                    except:
                        raise GrammarError("Line %i, Rule %i: Error near end of line." % (line_count, rule_count))

                    try:
                        lhs, rhsstring = content.split("->")
                    except:
                        raise GrammarError("Line %i, Rule %i: Invalid rule format." % (line_count, rule_count))
                    lhs = lhs.strip()
                    if rule_count == 1:
                        output.start_symbol = lhs
                    if "|" in rhsstring:
                        if not is_synchronous and rule_count > 1:
                            raise GrammarError("Line %i, Rule %i: All or none of the rules need to have two RHSs."
                                               % (line_count, rule_count))
                        is_synchronous = True
                        try:
                            rhs1, rhs2 = rhsstring.split("|")
                        except:
                            raise GrammarError("Only up to two RHSs are allowed in grammar file.")
                    else:
                        if is_synchronous and rule_count > 0:
                            raise ParserError("Line %i, Rule %i: All or none of the rules need to have two RHSs."
                                              % (line_count, rule_count))
                        is_synchronous = False
                        rhs1 = rhsstring
                        rhs2 = None

                    try:  # If the first graph in the file cannot be parsed, assume it's a string
                        r1 = Hgraph.from_string(rhs1)
                        r1_nts = set([(ntlabel.label, ntlabel.index) for h, ntlabel, t in r1.nonterminal_edges()])
                        if not rhs1_type:
                            rhs1_type = GRAPH_FORMAT
                    except (ParserError, IndexError) as e:
                        if rhs1_type == GRAPH_FORMAT:
                            raise ParserError("Line %i, Rule %i: Could not parse graph description: %s"
                                              % (line_count, rule_count, e.message))
                        else:
                            r1 = parse_string(rhs1)
                            nts = [t for t in r1 if isinstance(t, NonterminalLabel)]
                            r1_nts = set([(ntlabel.label, ntlabel.index) for ntlabel in nts])
                            rhs1_type = STRING_FORMAT

                    if is_synchronous:
                        try:  # If the first graph in the file cannot be parsed, assume it's a string
                            if rhs2_type:
                                assert rhs2_type == GRAPH_FORMAT
                            r2 = Hgraph.from_string(rhs2)
                            r2_nts = set([(ntlabel.label, ntlabel.index) for h, ntlabel, t in r2.nonterminal_edges()])
                            if not rhs2_type:
                                rhs2_type = GRAPH_FORMAT
                        except (ParserError, IndexError, AssertionError) as e:
                            if rhs2_type == GRAPH_FORMAT:
                                raise ParserError("Line %i, Rule %i: Could not parse graph description: %s"
                                                  % (line_count, rule_count, e))
                            else:
                                r2 = parse_string(rhs2)
                                nts = [t for t in r2 if isinstance(t, NonterminalLabel)]
                                r2_nts = set([(ntlabel.label, ntlabel.index) for ntlabel in nts])
                                rhs2_type = STRING_FORMAT
                        if not r1_nts == r2_nts:
                            raise GrammarError("Line %i, Rule %i: Nonterminals do not match between RHSs: %s %s"
                                               % (line_count, rule_count, str(r1_nts), str(r2_nts)))
                    else:
                        r2 = None
                    try:
                        if is_synchronous and reverse:
                            output[rule_count] = rule_class(rule_count, lhs, weight, r2, r1, nodelabels=nodelabels,
                                                            logprob=logprob)
                        else:
                            output[rule_count] = rule_class(rule_count, lhs, weight, r1, r2, nodelabels=nodelabels,
                                                            logprob=logprob)
                    except Exception as e:
                        raise GrammarError(
                            "Line %i, Rule %i: Could not initialize rule. %s" % (line_count, rule_count, e))
                    buf = StringIO()
                    rule_count += 1

        output.is_synchronous = is_synchronous
        if is_synchronous and reverse:
            output.rhs1_type, output.rhs2_type = rhs2_type, rhs1_type
        else:
            output.rhs1_type, output.rhs2_type = rhs1_type, rhs2_type

        output._compute_reachability_table_lookup()
        return output

    def _compute_reachability_table_lookup(self):
        """
        Fill a table mapping rhs symbols to rules so that we can compute reachability.
        """
        for r in self:
            rule = self[r]
            if self.rhs1_type is GRAPH_FORMAT:
                self.lhs_to_rules[rule.symbol, len(rule.rhs1.external_nodes)].add(r)
                terminals, nonterminals = rule.rhs1.get_terminals_and_nonterminals(self.nodelabels)
                for nt in nonterminals:
                    self.nonterminal_to_rules[nt].add(r)
            elif self.rhs1_type is STRING_FORMAT:
                terminals, nonterminals = _terminals_and_nts_from_string(rule.rhs1)
                self.lhs_to_rules[rule.symbol].add(r)
                for t in nonterminals:
                    self.nonterminal_to_rules[t].add(r)

    def terminal_filter(self, input1, input2):

        input1_terminals = set()
        input2_terminals = set()

        if self.rhs1_type is GRAPH_FORMAT:
            input1_terminals, input1_nts = input1.get_terminals_and_nonterminals(self.nodelabels)
        elif self.rhs1_type is STRING_FORMAT:
            input1_terminals, input1_nonterminals = _terminals_and_nts_from_string(input1)

        if input2:
            if self.rhs2_type is GRAPH_FORMAT:
                input2_terminals, input2_nts = input2.get_terminals_and_nonterminals(self.nodelabels)
            elif self.rhs2_type is STRING_FORMAT:
                input2_terminals, input2_nts = _terminals_and_nts_from_string(input2)

        accepted = list()
        for r in self:
            terminals1, terminals2 = set(), set()
            if self.rhs1_type is GRAPH_FORMAT:
                terminals1, nonterminals = self[r].rhs1.get_terminals_and_nonterminals(self.nodelabels)
            elif self.rhs1_type is STRING_FORMAT:
                terminals1, nonterminals = _terminals_and_nts_from_string(self[r].rhs1)
            if input2:
                if self.rhs2_type is GRAPH_FORMAT:
                    terminals2, nonterminals = self[r].rhs2.get_terminals_and_nonterminals(self.nodelabels)
                elif self.rhs2_type is STRING_FORMAT:
                    terminals2, nonterminals = _terminals_and_nts_from_string(self[r].rhs2)

            if terminals1.issubset(input1_terminals):
                if input2 is None or terminals2.issubset(input2_terminals):
                    accepted.append(r)
            if not terminals1 and not terminals2:
                accepted.append(r)

        return accepted

    def reachable_rules(self, input1, input2):
        todo = list(self.terminal_filter(input1, input2))
        result = set()
        while todo:
            r = todo.pop()
            result.add(r)
            todo.extend(self.nonterminal_to_rules[self[r].symbol] - result)
        return result
