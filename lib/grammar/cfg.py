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

# class ExtractorCFG(CFG):
#
#     def __init__(self):
#         super(ExtractorCFG, self).__init__()
#         self.parse()
#         for prod in self.prods:
#             self.name_to_prod[prod.operator_name] = prod
#             for sym in prod.argument_symbols:
#                 if not prod in self.symbol_to_out_productions[sym]:
#                     self.symbol_to_out_productions[sym].append(prod)
#         ei_sym: NonterminalSymbol = self.name_to_sym["ei"]
#         for prod in ei_sym.prods:
#             if prod.heuristics is not None:
#                 for heuristic in prod.heuristics:
#                     self.ei_symbol_heursitics[str(heuristic)].append(prod)
#
#         print(self.ei_symbol_heursitics)
#         print(self.symbol_to_out_productions)
#
#     def parse(self):
#         # creating the symbols
#         # nonterminals
#         nn_sym = NonterminalSymbol("nn")
#         en_sym = NonterminalSymbol("en")
#         en_1_sym = NonterminalSymbol("en_1")
#         pred_en_1_sym = NonterminalSymbol("pred_en_1")
#         pred_en_2_sym = NonterminalSymbol("pred_en_2")
#         pred_en_3_sym = NonterminalSymbol("pred_en_3")
#         pred_nn_sym = NonterminalSymbol("pred_nn")
#         sn_sym = NonterminalSymbol("sn")
#
#         ei_sym = NonterminalSymbol("ei")
#         p_sym = NonterminalSymbol("p", max_sym_depth=1)
#
#         self.name_to_sym["ei"] = ei_sym
#         self.start_sym = ei_sym
#
#         # terminals
#         input_sym = TerminalSymbol("input")
#         k_sym = TerminalSymbol("k", values=K)
#         keyword_sym = TerminalSymbol("keyword")
#         const_str_sym = TerminalSymbol("const_str")
#         q_sym = TerminalSymbol("q")
#         label_sym = TerminalSymbol("label", values=get_entity())
#         threshold_sym = TerminalSymbol("threshold", values=THRESHOLD)
#
#         eps_sym = TerminalSymbol("eps")
#
#         # productions:
#         nn_productions = [Production(nn_sym, [en_1_sym], "toNN"),
#                           Production(nn_sym, [en_1_sym], "getLeaves-M"),
#                           Production(nn_sym, [en_sym, pred_nn_sym], "getChildren"),
#                           Production(nn_sym, [en_sym, pred_nn_sym], "getLeaves")]
#
#         nn_sym.prods = nn_productions
#         self.prods.extend(nn_productions)
#
#         pred_nn_productions = [Production(pred_nn_sym, [], "isList"),
#                             #    Production(pred_nn_sym, [], "isString"),
#                                Production(pred_nn_sym, [], "isAny")]
#
#         pred_nn_sym.prods = pred_nn_productions
#         self.prods.extend(pred_nn_productions)
#
#         en_productions = [
#             Production(en_sym, [pred_en_1_sym, input_sym, k_sym], "getNode"),
#             Production(en_sym, [pred_en_2_sym, input_sym, k_sym, threshold_sym], "getNode-T"),
#             # Production(en_sym, [pred_en_3_sym, input_sym], "getNode-M")
#         ]
#
#         en_sym.prods = en_productions
#         self.prods.extend(en_productions)
#
#         en_1_productions = [
#             Production(en_1_sym, [pred_en_3_sym, input_sym], "getNode-M")
#         ]
#
#         en_1_sym.prods = en_1_productions
#         self.prods.extend(en_1_productions)
#
#         pred_en_1_productions = [Production(pred_en_1_sym, [keyword_sym], "matchKeyword"),
#                                Production(pred_en_1_sym, [q_sym], "matchQA"),
#                                Production(pred_en_1_sym, [q_sym, keyword_sym], "matchSection1"),
#                             #    Production(pred_en_sym, [q_sym, keyword_sym], "matchSection2"),
#                             #    Production(pred_en_1_sym, [], "matchDoc")]
#         ]
#
#         pred_en_1_sym.prods = pred_en_1_productions
#         self.prods.extend(pred_en_1_productions)
#         self.base_case_init_sym.append(pred_en_1_sym)
#
#         pred_en_2_productions = [Production(pred_en_2_sym, [q_sym, keyword_sym], "matchSection2")]
#
#         pred_en_2_sym.prods = pred_en_2_productions
#         self.prods.extend(pred_en_2_productions)
#         self.base_case_init_sym.append(pred_en_2_sym)
#
#         pred_en_3_productions = [Production(pred_en_3_sym, [eps_sym], "matchDoc")]
#
#         pred_en_3_sym.prods = pred_en_3_productions
#         self.prods.extend(pred_en_3_productions)
#         self.base_case_init_sym.append(pred_en_3_sym)
#
#         sn_productions = [Production(sn_sym, [nn_sym], "nodeToStr")]
#
#         sn_sym.prods = sn_productions
#         self.prods.extend(sn_productions)
#
#         ei_productions = [Production(ei_sym, [sn_sym], "toEI"),
#                           Production(ei_sym, [ei_sym, p_sym], "filter", [["toEI"],
#                                                                          ["getAnswer", "toEI"],
#                                                                          ["getCSVItem", "toEI"]
#                                                                          ]),
#                           Production(ei_sym, [ei_sym, label_sym], "getEntity", [["toEI"],
#                                                                                 ["getCSVItem", "toEI"],
#                                                                                 ["getCSVItem", "filter", "toEI"],
#                                                                                 ["filter", "toEI"],
#                                                                                 ["filter", "getAnswer", "toEI"],
#                                                                                 ["filter", "getCSVItem", "toEI"]
#                                                                                 ]),
#                           Production(ei_sym, [ei_sym], "getCSVItem", [["toEI"],
#                                                                       ["filter", "toEI"]]),
#                           Production(ei_sym, [ei_sym, const_str_sym, threshold_sym], "getString", [["toEI"],
#                                                                                                    ["getAnswer",
#                                                                                                     "toEI"],
#                                                                                                    ["getCSVItem",
#                                                                                                     "toEI"],
#                                                                                                    ["getCSVItem",
#                                                                                                     "filter", "toEI"]]),
#                           Production(ei_sym, [ei_sym, q_sym, k_sym], "getAnswer", [["toEI"],
#                                                                                    ["filter", "toEI"]])
#                           ]
#
#         ei_sym.prods = ei_productions
#         self.prods.extend(ei_productions)
#
#         p_productions = [
#             Production(p_sym, [p_sym], "_not", is_lambda=True, is_recursive=True),
#             Production(p_sym, [p_sym, p_sym], "_or", is_lambda=True, is_recursive=True),
#             Production(p_sym, [p_sym, p_sym], "_and", is_lambda=True, is_recursive=True),
#             Production(p_sym, [label_sym], "hasEntity", is_lambda=True),
#             Production(p_sym, [const_str_sym, threshold_sym], "hasString", is_lambda=True),
#             # Production(p_sym, [], "true", lambda x: True, is_lambda=True)
#         ]
#
#         p_sym.prods = p_productions
#         self.prods.extend(p_productions)
#
#
# class PredicateCFG(CFG):
#
#     def __init__(self):
#         super(PredicateCFG, self).__init__()
#         self.parse()
#         for prod in self.prods:
#             self.name_to_prod[prod.operator_name] = prod
#
#     def parse(self):
#         input_sym = TerminalSymbol("input")
#         program_sym = TerminalSymbol("program")
#         k_sym = TerminalSymbol("k", values=[1])
#         keyword_sym = TerminalSymbol("keyword")
#         const_str_sym = TerminalSymbol("const_str")
#         q_sym = TerminalSymbol("q")
#         label_sym = TerminalSymbol("label", values=get_pred_entity())
#         threshold_sym = TerminalSymbol("threshold", values=THRESHOLD)
#         eps_sym = TerminalSymbol("eps")
#
#         t_sym = NonterminalSymbol("t")
#         pred_lt_1_sym = NonterminalSymbol("pred_lt_1")
#         pred_lt_2_sym = NonterminalSymbol("pred_lt_2")
#         pred_sym = NonterminalSymbol("pred", max_sym_depth=2)
#         pt_sym = NonterminalSymbol("pt")
#         d_sym = NonterminalSymbol("d")
#         d_1_sym = NonterminalSymbol("d_1")
#         d_2_sym = NonterminalSymbol("d_2")
#
#         self.start_sym = pred_sym
#
#         t_productions = [
#             Production(t_sym, [pred_lt_1_sym, input_sym, k_sym], "getNode", prefix="preddsl"),
#             Production(t_sym, [pred_lt_2_sym, input_sym, k_sym, threshold_sym], "getNode-T", prefix="preddsl")
#         ]
#         t_sym.prods = t_productions
#         self.prods.extend(t_productions)
#
#         pred_lt_1_productions = [
#             Production(pred_lt_1_sym, [keyword_sym], "matchKeyword", prefix="preddsl"),
#             Production(pred_lt_1_sym, [q_sym], "matchQA", prefix="preddsl"),
#             Production(pred_lt_1_sym, [q_sym, keyword_sym], "matchSection1", prefix="preddsl"),
#             Production(pred_lt_1_sym, [eps_sym], "matchDoc", prefix="preddsl")
#         ]
#         pred_lt_1_sym.prods = pred_lt_1_productions
#         self.prods.extend(pred_lt_1_productions)
#
#         pred_lt_2_productions = [
#             Production(pred_lt_2_sym, [q_sym, keyword_sym], "matchSection2", prefix="preddsl")
#         ]
#         pred_lt_2_sym.prods = pred_lt_2_productions
#         self.prods.extend(pred_lt_2_productions)
#
#         pred_productions = [
#             Production(pred_sym, [pt_sym, d_sym], "contains", prefix="preddsl"),
#             Production(pred_sym, [pt_sym, d_1_sym, d_2_sym], "contains-D", prefix="preddsl"),
#             Production(pred_sym, [t_sym], "containsList", prefix="preddsl"),
#             Production(pred_sym, [pt_sym], "isSingleton", prefix="preddsl"),
#             Production(pred_sym, [pred_sym], "_not", prefix="preddsl"),
#             # Production(pred_sym, [pred_sym, pred_sym], "_and"),
#             # Production(pred_sym, [input_sym, program_sym], "isEmpty"),  # TBD
#             Production(pred_sym, [threshold_sym], "headerConfidence", prefix="preddsl")
#         ]
#         pred_sym.prods = pred_productions
#         self.prods.extend(pred_productions)
#
#         pt_productions = [
#             Production(pt_sym, [t_sym], "leaves", prefix="preddsl"),
#             # Production(pt_sym, [t_sym], "children"),
#             Production(pt_sym, [t_sym], "descendents", prefix="preddsl"),
#         ]
#         pt_sym.prods = pt_productions
#         self.prods.extend(pt_productions)
#
#         d_productions = [
#             Production(d_sym, [const_str_sym], "toConst", prefix="preddsl"),
#             Production(d_sym, [label_sym], "toLabel", prefix="preddsl"),
#             Production(d_sym, [keyword_sym], "toKeyword", prefix="preddsl")
#         ]
#         d_sym.prods = d_productions
#         self.prods.extend(d_productions)
#
#         d_1_productions = [
#             Production(d_sym, [const_str_sym], "toConst", prefix="preddsl"),
#             # Production(d_sym, [label_sym], "toLabel", prefix="preddsl"),
#             # Production(d_sym, [keyword_sym], "toKeyword", prefix="preddsl")
#         ]
#         d_1_sym.prods = d_1_productions
#         # self.prods.extend(d_productions)
#
#         d_2_productions = [
#             # Production(d_sym, [const_str_sym], "toConst", prefix="preddsl"),
#             Production(d_sym, [label_sym], "toLabel", prefix="preddsl"),
#             # Production(d_sym, [keyword_sym], "toKeyword", prefix="preddsl")
#         ]
#         d_2_sym.prods = d_2_productions
#         # self.prods.extend(d_productions)
