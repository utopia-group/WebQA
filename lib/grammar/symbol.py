class Symbol:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class TerminalSymbol(Symbol):
    def __init__(self, name: str, values=None):
        super(TerminalSymbol, self).__init__(name)
        self.values = values

    def get_cost(self, value):
        return 0


class NonterminalSymbol(Symbol):
    def __init__(self, name: str, max_sym_depth=-1, is_recursive=False):
        super(NonterminalSymbol, self).__init__(name)
        # NOTE: we don't do type hints because this leads to circular import issue
        self.prods = []
        self.max_sym_depth = max_sym_depth
        self.is_recursive = is_recursive
