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


class CkyTooLongException(Exception):
    def __init__(self, steps):
        self.steps = steps
