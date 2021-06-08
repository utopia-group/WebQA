from lib.grammar.symbol import Symbol, NonterminalSymbol
from typing import List, Callable


class Production:
    def __init__(self, return_symbol: NonterminalSymbol, argument_symbols: List[Symbol], operator_name: str,
                 heuristics: List = None, cost: float = 1, is_lambda: bool = False, is_recursive: bool = False, prefix="dsl"):
        self.return_symbol: NonterminalSymbol = return_symbol
        self.argument_symbols: List[Symbol] = argument_symbols
        self.operator_name: str = operator_name

        self.heuristics = heuristics
        self.cost = cost
        self.is_lambda = is_lambda
        self.is_recursive = any([arg.name == self.return_symbol.name for arg in self.argument_symbols])

        self.prefix=prefix

    def repr_helper(self, args: List[str]):

        if self.is_lambda:
            processed_args = []
            for arg in args:
                if arg.startswith("lambda"):
                    processed_args.append("{} x".format(arg))
                else:
                    processed_args.append(arg)
            return "lambda x: {}({})".format(self.getName(), ", ".join(processed_args))
        return "{}({})".format(self.getName(), ", ".join(args))

    def exec_helper(self, args: List[str], exclude_prefix):
        if self.is_lambda:
            processed_args = []
            for arg in args:
                # if arg.startswith("lambda"):
                #     processed_args.append("{}.{} x".format(self.prefix, arg))
                # else:
                #     processed_args.append(arg)
                processed_args.append(arg)

            # NOTE: generate the lambda var for hasString and hasEnt
            if self.getName().startswith("has"):
                processed_args.insert(0, "x")

            if exclude_prefix:
                return "lambda x: self.{}({})".format(self.getName(), ", ".join(processed_args))
            else:
                return "lambda x: {}.{}({})".format(self.prefix, self.getName(), ", ".join(processed_args))
        if self.return_symbol.name.startswith("np") or self.return_symbol.name == "pn":
            return "{}.{}".format(self.prefix, self.getName())

        return "{}.{}({})".format(self.prefix, self.getName(), ", ".join(args))

    def getName(self):
        if "-" not in self.operator_name:
            return self.operator_name
        else:
            return self.operator_name.split("-")[0]

    def __repr__(self):
        return self.operator_name
