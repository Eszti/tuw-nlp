class InputFormatException(Exception):
    pass


class BinarizationException(Exception):
    pass


class DerivationException(Exception):
    pass


class LexerError(Exception):
    pass


class ParserError(Exception):
    pass


class GrammarError(Exception):
    pass


class ParseTooLongException(Exception):
    def __init__(self, steps, queue, attempted):
        self.steps = steps
        self.queue_len = len(queue)
        self.attempted_len = len(attempted)

    def print_message(self):
        return f"Parse did not finish:\n" \
               f"- steps: {self.steps}\n" \
               f"- queue len: {self.queue_len}\n" \
               f"- attempted len:{self.attempted_len}\n"


class CkyTooLongException(Exception):
    def __init__(self, steps):
        self.steps = steps

    def print_message(self):
        return f"Cky conversion did not finish:\n- steps: {self.steps}\n"


class NotAllNodesCoveredException(Exception):
    def __init__(self, orig_nodes, derived_nodes, not_covered_nodes):
        self.orig_nodes = orig_nodes
        self.derived_nodes = derived_nodes
        self.not_covered_nodes = not_covered_nodes

    def print_message(self):
        return f"{len(self.derived_nodes)}/{len(self.not_covered_nodes)} " \
               f"(covered/not covered) out of {len(self.orig_nodes)}\n"\
               f"Not covered nodes:\n{self.not_covered_nodes}\n"
