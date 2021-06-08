from lib.grammar.symbol import *
from lib.grammar.production import *
from lib.grammar.constant import *
from typing import List, Dict, Tuple
from collections import defaultdict


class CFG:
    def __init__(self):
        self.terminal_syms: List[TerminalSymbol] = []
        self.nonterminal_syms: List[NonterminalSymbol] = []
        self.start_sym: NonterminalSymbol = None
        self.base_case_init_sym: List[NonterminalSymbol] = []

        self.name_to_sym: Dict[str, Symbol] = {}
        self.name_to_prod: Dict[str, Production] = {}

        self.symbol_to_out_productions: defaultdict[Symbol, List[Production]] = defaultdict(list)

        self.prods: List[Production] = []

    def parse(self):
        raise NotImplementedError


class ExtractorCFG(CFG):
    def __init__(self):
        super(ExtractorCFG, self).__init__()
        self.parse()
        self.e_symbol_heursitics: defaultdict[Tuple, List[Production]] = defaultdict(list)

        for prod in self.prods:
            self.name_to_prod[prod.operator_name] = prod
            for sym in prod.argument_symbols:
                if not prod in self.symbol_to_out_productions[sym]:
                    self.symbol_to_out_productions[sym].append(prod)

        e_sym: NonterminalSymbol = self.name_to_sym["e"]
        for prod in e_sym.prods:
            if prod.heuristics is not None:
                for heuristic in prod.heuristics:
                    self.e_symbol_heursitics[str(heuristic)].append(prod)

        # print(self.e_symbol_heursitics)
        # print(self.symbol_to_out_productions)

    def parse(self):
        e_sym = NonterminalSymbol('e')
        self.name_to_sym['e'] = e_sym
        np_2_sym = NonterminalSymbol('np_2')
        self.name_to_sym['np_2'] = np_2_sym

        self.start_sym = e_sym

        # terminals
        x_sym = TerminalSymbol("x")
        k_sym = TerminalSymbol("k", values=K)
        const_str_sym = TerminalSymbol("const_str")
        q_sym = TerminalSymbol("q")
        label_sym = TerminalSymbol("label", values=get_entity())
        threshold_sym = TerminalSymbol("threshold", values=THRESHOLD)

        e_sym_productions = [
            Production(e_sym, [x_sym], "ExtractContent"),
            Production(e_sym, [e_sym, np_2_sym], "Filter", [["ExtractContent"],
                                                            ["GetAnswer", "ExtractContent"],
                                                            ["Split", "ExtractContent"]]),

            Production(e_sym, [e_sym, label_sym], "GetEntity", [["ExtractContent"],
                                                                ["Split", "ExtractContent"],
                                                                ["Filter", "ExtractContent"],
                                                                ["Filter", "GetAnswer", "ExtractContent"],
                                                                # ["Filter", "Split", "ExtractContent"]
                                                                ]),
            Production(e_sym, [e_sym], "Split", [["ExtractContent"],
                                                 ["Filter", "ExtractContent"]]),
            Production(e_sym, [e_sym, const_str_sym, threshold_sym], "GetString", [["ExtractContent"],
                                                                                   ["GetAnswer", "ExtractContent"],
                                                                                   ["Split", "ExtractContent"],
                                                                                   # ["Split", "Filter",
                                                                                   #  "ExtractContent"]
                                                                                   ]),
            Production(e_sym, [e_sym, q_sym, k_sym], "GetAnswer", [["ExtractContent"],
                                                                   ["Filter", "ExtractContent"]
                                                                   ]),

        ]
        e_sym.prods = e_sym_productions
        self.prods.extend(e_sym_productions)

        np_2_sym_productions = [
            # Production(np_2_sym, [q_sym], "hasAnswer", is_lambda=True),
            Production(np_2_sym, [const_str_sym, threshold_sym], "hasString", is_lambda=True),
            Production(np_2_sym, [label_sym], "hasEntity", is_lambda=True),
        ]

        np_2_sym.prods = np_2_sym_productions
        self.prods.extend(np_2_sym_productions)


class GuardCFG(CFG):

    def __init__(self):
        super(GuardCFG, self).__init__()
        self.parse()

        # init for bottom-up related stuff
        for prod in self.prods:
            self.name_to_prod[prod.operator_name] = prod
            for sym in prod.argument_symbols:
                if not prod in self.symbol_to_out_productions[sym]:
                    self.symbol_to_out_productions[sym].append(prod)
        # print(self.symbol_to_out_productions)

    def parse(self):
        g_sym = NonterminalSymbol('g')
        v_sym = NonterminalSymbol('v')
        n_sym = NonterminalSymbol('n')
        n_1_sym = NonterminalSymbol('n_1')
        pn_sym = NonterminalSymbol('pn')
        np_1_1_sym = NonterminalSymbol('np_1_1')
        np_1_2_sym = NonterminalSymbol('np_1_2')
        np_1_3_sym = NonterminalSymbol('np_1_3')
        np_2_sym = NonterminalSymbol('np_2', max_sym_depth=2)
        self.name_to_sym['np_2'] = np_2_sym

        self.base_case_init_sym.append(np_1_1_sym)
        self.base_case_init_sym.append(np_1_2_sym)
        self.base_case_init_sym.append(np_1_3_sym)
        self.base_case_init_sym.append(n_1_sym)

        # terminals
        x_sym = TerminalSymbol("x")
        w_sym = TerminalSymbol("w")
        k_sym = TerminalSymbol("k", values=K)
        keyword_sym = TerminalSymbol("keyword")
        const_str_sym = TerminalSymbol("const_str")
        q_sym = TerminalSymbol("q")
        label_sym = TerminalSymbol("label", values=get_entity())
        threshold_sym = TerminalSymbol("threshold", values=THRESHOLD)
        threshold_2_sym = TerminalSymbol("threshold_2", values=THRESHOLD_2)
        threshold_3_sym = TerminalSymbol("threshold_3", values=THRESHOLD_3)

        # productions
        g_sym_productions = [
            Production(g_sym, [v_sym, np_2_sym], "AnySat"),
            Production(g_sym, [v_sym], "IsSingleton"),
        ]
        g_sym.prods = g_sym_productions
        self.prods.extend(g_sym_productions)

        v_sym_productions = [
            Production(v_sym, [n_1_sym], "toV"),
            Production(v_sym, [n_1_sym], "GetLeaves-M"),
            Production(v_sym, [n_sym, pn_sym], "GetChildren"),
            Production(v_sym, [n_sym, pn_sym], "GetLeaves"),
        ]

        v_sym.prods = v_sym_productions
        self.prods.extend(v_sym_productions)

        n_sym_productions = [
            Production(n_sym, [np_1_1_sym, w_sym, k_sym], "GetNode"),
            Production(n_sym, [np_1_2_sym, w_sym, k_sym, threshold_2_sym], "GetNode-T2"),
            Production(n_sym, [np_1_3_sym, w_sym, k_sym, threshold_sym], "GetNode-T"),
        ]
        n_sym.prods = n_sym_productions
        self.prods.extend(n_sym_productions)

        n_1_sym_productions = [
            Production(n_1_sym, [w_sym], "GetRoot")
        ]
        n_1_sym.prods = n_1_sym_productions
        self.prods.extend(n_1_sym_productions)

        pn_sym_productions = [
            Production(pn_sym, [], "isStructured"),
            Production(pn_sym, [], "isAny")
        ]
        pn_sym.prods = pn_sym_productions
        self.prods.extend(pn_sym_productions)

        np_1_1_sym_productions = [
            Production(np_1_1_sym, [q_sym], "matchQA"),
        ]
        np_1_1_sym.prods = np_1_1_sym_productions
        self.prods.extend(np_1_1_sym_productions)

        np_1_2_sym_productions = [
            Production(np_1_2_sym, [keyword_sym], "matchKeyword"),
            Production(np_1_2_sym, [q_sym, keyword_sym], "matchSection1"),
        ]
        np_1_2_sym.prods = np_1_2_sym_productions
        self.prods.extend(np_1_2_sym_productions)

        np_1_3_sym_productions = [
            Production(np_1_3_sym, [q_sym, keyword_sym], "matchSection2"),
        ]
        np_1_3_sym.prods = np_1_3_sym_productions
        self.prods.extend(np_1_3_sym_productions)

        np_2_sym_productions = [
            Production(np_2_sym, [x_sym], "_true", is_lambda=True),
            Production(np_2_sym, [threshold_sym], "hasHeader", is_lambda=True),
            Production(np_2_sym, [const_str_sym, threshold_3_sym], "hasString", is_lambda=True),
            Production(np_2_sym, [label_sym], "hasEntity", is_lambda=True),
            Production(np_2_sym, [const_str_sym, label_sym, threshold_3_sym], "hasStrEnt", is_lambda=True),
            Production(np_2_sym, [np_2_sym], "_not", is_lambda=True),
        ]

        np_2_sym.prods = np_2_sym_productions
        self.prods.extend(np_2_sym_productions)

        self.start_sym = g_sym
