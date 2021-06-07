import itertools
from typing import List, Tuple

from lib.evaluator import Evaluator
from lib.grammar.cfg import CFG, ExtractorCFG, GuardCFG
from lib.grammar.production import Production
from lib.grammar.symbol import TerminalSymbol, NonterminalSymbol, Symbol
from lib.interpreter.context import SynthProgContext, SynthProgContextExt, Subset
from lib.program import Program, NonterminalNode, GuardProgram, IEProgram, LocatorProgram
from lib.spec import ExtractExample, Task, Example, PredExample
from lib.utils.pq import PriorityQueue
from lib.utils.misc_utils import get_priority_f1_reverse, get_priority_depth_reverse


class Synthesizer:
    def __init__(self, grammar, size_limit=100, random_sample=False):
        self.grammar: CFG = grammar
        self.evaluator: Evaluator = Evaluator()
        self.size_limit = size_limit
        self.sample = random_sample
        self.id_counter = itertools.count(start=1)

    def synthesize_best(self, task: Task, examples: List[Example], enum_limit, depth_lim, pruning) \
            -> List[SynthProgContext]:
        raise NotImplementedError

    def synthesize(self, task: Task, examples: List[Example], enum_limit, depth_lim, pruning, optim_f1) -> \
            Tuple[List[SynthProgContext], float]:
        raise NotImplementedError

    def init_prog(self, task: Task, examples: List[Example] = None, pruning=False):
        raise NotImplementedError

    def eval_prog(self, task: Task, examples: List[Example], curr_p: Program):
        raise NotImplementedError

    def infer_applicable_prods_bot_up(self, prog: Program, sym: Symbol, parent_node_id: int):
        if sym.name == "v":
            return []
        if sym.name == "e":
            assert isinstance(self.grammar, ExtractorCFG)
            key = []
            curr_node = prog.get_node(parent_node_id)
            for i in range(3):
                if curr_node is None or not curr_node.sym.name == "e":
                    break
                assert isinstance(curr_node, NonterminalNode)
                prod_name = "toEI" if curr_node.prod.operator_name == "nodeToStr" else \
                    curr_node.prod.operator_name
                key.append(prod_name)
                curr_node = prog.get_node(prog.get_children(curr_node)[0])
            assert len(key) > 0
            # print("key:", key)
            applicable_prods = self.grammar.e_symbol_heursitics[str(key)]
            # print("applicable_prods:", applicable_prods)
        else:
            applicable_prods = self.grammar.symbol_to_out_productions[sym]

        return applicable_prods

    def infer_applicable_prods_top_down(self, prog: Program, sym: Symbol, parent_node_id: int):

        assert isinstance(sym, NonterminalSymbol)

        recorded_depth = -1
        if sym.max_sym_depth > -1:
            if prog is None:
                recorded_depth = 0
            else:
                parent_node = prog.get_node(parent_node_id)
                if parent_node is None:
                    recorded_depth = 0
                elif not parent_node.sym.name == sym.name:
                    recorded_depth = 0
                else:
                    assert isinstance(parent_node, NonterminalNode)
                    recorded_depth = parent_node.sym_depth
            if (recorded_depth + 1) >= sym.max_sym_depth:
                applicable_prods = [prod for prod in sym.prods if not prod.is_recursive]
            else:
                applicable_prods = sym.prods
            recorded_depth += 1
        else:
            applicable_prods = sym.prods

        assert len(applicable_prods) > 0

        return recorded_depth, applicable_prods


class BottomUpSynthesizer(Synthesizer):
    def __init__(self, grammar, size_limit=100, random_sample=False):
        super(BottomUpSynthesizer, self).__init__(grammar, size_limit, random_sample)
        # self.worklist = PriorityQueue(priority=get_priority_depth_reverse)
        self.worklist = PriorityQueue(priority=get_priority_f1_reverse)
        self.total_explored_states = 0

    def synthesize(self, task: Task, examples: List[Example], enum_limit, depth_lim, pruning, optim_f1,
                   locator_prog_id=None, decomposition=True) \
            -> Tuple[List[SynthProgContext], float]:

        total_states_pull = 0

        optim_programs = []
        optim_f1 = optim_f1

        best_section_matching_recall = 0.0
        best_section_matching_prod_name = []

        enumerated_programs = []

        # we need to exec to init prog here (because they are already complete)
        # TODO: this can definitely be optimized
        init_progs = self.init_prog(task, examples)
        for prog in init_progs:
            if prog.start_node.sym == self.grammar.start_sym and isinstance(prog.start_node, NonterminalNode) and \
                    prog.start_node.prod.operator_name == "ExtractContent":
                prog_exec_context = self.evaluator.eval_prod(task, examples, prog,
                                                             self.grammar.name_to_prod['ExtractContent'],
                                                             context_format=True, pruning=pruning,
                                                             locator_prog_id=locator_prog_id)

                avg_f1 = prog_exec_context.get_stats('f1')
                if avg_f1 > optim_f1:
                    optim_f1 = avg_f1
                    print("update optim_f1: ", optim_f1)
                    optim_programs = [prog_exec_context]
                elif avg_f1 == optim_f1:
                    optim_programs.append(prog_exec_context)

                if prog.depth <= (depth_lim - 1):
                    if not pruning:
                        self.worklist.put(prog)
                    if pruning and prog.get_avg_f1() >= optim_f1 and not prog.check_empty_f1(approx=False):
                        self.worklist.put(prog)

        while not self.worklist.is_empty():

            curr_p: Program = self.worklist.pop()
            curr_p_str = str(curr_p)
            total_states_pull += 1
            # print("{} curr_p: {}".format(total_states_pull, curr_p_str))
            # print("curr_p.curr_sym:", curr_p.curr_sym)
            # print("curr_p depth:", curr_p.depth)

            if len(curr_p.nodes) > self.size_limit:
                break
            if curr_p.depth >= depth_lim:
                # print("reach depth limit")
                continue
            if len(enumerated_programs) > enum_limit:
                # print("reach enum limit")
                return optim_programs, optim_f1

            assert isinstance(curr_p.curr_sym, NonterminalSymbol)

            for prod in self.infer_applicable_prods_bot_up(curr_p, curr_p.curr_sym, curr_p.start_node.id):
                # print("prod:", prod.operator_name)
                new_progs: List[Program] = self.expand_prog_with_production(curr_p, curr_p.curr_sym, prod)

                for prog in new_progs:

                    if prod.return_symbol == self.grammar.start_sym:

                        prog_exec_context: SynthProgContextExt = self.evaluator.eval_prod(task, examples, prog, prod,
                                                                                          context_format=True,
                                                                                          pruning=pruning,
                                                                                          locator_prog_id=locator_prog_id)
                        avg_f1 = prog_exec_context.get_stats('f1')
                        if avg_f1 > optim_f1:
                            optim_f1 = avg_f1
                            print("update optim_f1: ", optim_f1)
                            if not prod.return_symbol.name == "nn":
                                optim_programs = [prog_exec_context]

                        elif avg_f1 == optim_f1:
                            if not prod.return_symbol.name == "nn":
                                optim_programs.append(prog_exec_context)

                        if curr_p.depth <= (depth_lim - 1):
                            if not pruning:
                                self.worklist.put(prog)
                            if pruning and prog.get_avg_f1() >= optim_f1 and not prog.check_empty_f1(approx=False):
                                # print("avg stats: {}, {}, {}".format(avg_f1, prog_exec_context.get_stats('precision'),
                                #                                      prog_exec_context.get_stats('recall')))
                                # print("approx stats: {}, {}, {}".format(prog.get_avg_f1(),
                                #                                         prog.get_avg_precision(),
                                #                                         prog.get_avg_recall()))
                                # print("detailed approx stats: {}".format(prog.approx_stats))
                                self.worklist.put(prog)
                        continue

                    if curr_p.depth > (depth_lim - 1):
                        continue

                    # only incomplete prog
                    assert (not prod.return_symbol == self.grammar.start_sym)
                    self.evaluator.eval_prod(task, examples, prog, prod, pruning=pruning,
                                             locator_prog_id=locator_prog_id)

                    if not pruning:
                        self.worklist.put(prog)
                    elif prog.get_avg_f1() >= optim_f1 and not prog.check_empty_f1(approx=False):
                        # print("avg stats: {}, {}, {}".format(prog.get_avg_f1(approx=False),
                        #                                      prog.get_avg_precision(approx=False),
                        #                                      prog.get_avg_recall(approx=False)))
                        # print("approx stats: {}, {}, {}".format(prog.get_avg_f1(),
                        #                                         prog.get_avg_precision(),
                        #                                         prog.get_avg_recall()))
                        # print("detailed approx stats: {}".format(prog.approx_stats))
                        self.worklist.put(prog)

        # print("total state pulled:", total_states_pull)
        self.total_explored_states += total_states_pull
        return optim_programs, optim_f1

    # NOTE: since section locators are never executed on neg benchmarks, we need to execute it here
    def expand_prog_with_production(self, prog: Program, curr_sym: Symbol, prod: Production) -> List[Program]:

        assert isinstance(curr_sym, NonterminalSymbol)

        # print("expand with production:", prod)

        ret_progs = []
        new_prog: Program = prog.duplicate(next(self.id_counter))

        # print("new prog:", new_prog)

        if prod.operator_name.startswith("to"):
            new_prog.curr_sym = prod.return_symbol
            new_prog.start_node.sym = prod.return_symbol
            return [new_prog]

        prev_start_node = new_prog.start_node
        new_node = new_prog.add_nonterminal_node(prod.return_symbol, prod, None, start_node=True, bot_up=True)
        new_prog.curr_sym = prod.return_symbol

        # NOTE: this is based on the assumption that the recursive symbol show first in all the operators' arguments
        self.set_parent_child(new_prog, new_node.id, prev_start_node.id)

        # NOTE: this part of the code is written based on the assumption that the recursive symbol will always only
        #  occur in the first argument
        mini_worklist = [new_prog]
        for arg_sym in prod.argument_symbols:
            # print("arg_sym:", arg_sym)
            # print("pred start node sym:", prev_start_node.sym)

            if arg_sym.name == prev_start_node.sym.name or arg_sym.name == "n_1":
                continue
            temp_progs = []
            for todo_prog in mini_worklist:
                temp_progs.extend(self.instantiate_prog_with_sym(todo_prog, arg_sym, new_node.id))
            mini_worklist = temp_progs

        ret_progs.extend(mini_worklist)
        return ret_progs

    def init_prog(self, task: Task, examples: List[Example] = None, pruning=False) -> List[IEProgram]:
        pass

    def eval_prog(self, task: Task, examples: List[Example], curr_p: Program):
        pass

    def synthesize_best(self, task: Task, examples: List[Example], enum_limit, depth_lim, pruning) \
            -> List[SynthProgContext]:
        pass

    def init_subprog(self, sym: Symbol) -> IEProgram:
        p = IEProgram(next(self.id_counter))
        p.curr_sym = sym
        return p

    def set_parent_child(self, prog: Program, parent_node_id: int, new_node_id: int):
        new_node = prog.get_node(new_node_id)
        prog.set_parent(prog.get_node(parent_node_id), new_node)
        if prog.to_children_edges.get(parent_node_id) is None:
            prog.to_children_edges[parent_node_id] = []
        else:
            # create a new list here, otherwise we need to use deepcopy for to_children_edges when duplicating programs
            prog.to_children_edges[parent_node_id] = [_id for _id in prog.to_children_edges[parent_node_id]]
        prog.to_children_edges[parent_node_id].append(new_node.id)

    # This function filled the arguments in a top-down manner
    def instantiate_prog_with_sym(self, p: IEProgram, sym: Symbol, parent_node_id: int, specific_prod=None) -> List[
        IEProgram]:

        return_progs = []

        if isinstance(sym, NonterminalSymbol):
            # compute applicable productions depends on recursive criteria
            # print("sym here:", sym)
            recorded_depth, applicable_prods = self.infer_applicable_prods_top_down(p, sym, parent_node_id) if \
                specific_prod is None else (0, specific_prod)

            for prod in applicable_prods:
                # print("prod:", prod)
                prog = self.init_subprog(sym) if p is None else p.duplicate(next(self.id_counter))
                new_node = prog.add_nonterminal_node(sym, prod, None) if not parent_node_id == -1 else \
                    prog.add_nonterminal_node(sym, prod, None, start_node=True)
                if parent_node_id == -1:
                    new_node.depth = 1
                if not parent_node_id == -1:
                    self.set_parent_child(prog, parent_node_id, new_node.id)
                new_node.sym_depth = recorded_depth

                if len(prod.argument_symbols) > 0:
                    mini_worklist = [prog]
                    for arg_sym in prod.argument_symbols:
                        temp_progs = []
                        for todo_prog in mini_worklist:
                            temp_progs.extend(self.instantiate_prog_with_sym(todo_prog, arg_sym, new_node.id))
                        mini_worklist = temp_progs
                    return_progs.extend(mini_worklist)
                else:
                    prog.set_children(new_node, [])
                    return_progs.append(prog)

        elif isinstance(sym, TerminalSymbol):
            values = sym.values if sym.values is not None else [sym.name]

            for value in values:
                prog = self.init_subprog(sym) if p is None else p.duplicate(next(self.id_counter))
                # print("before prog:", prog)
                # print("p children:", p.to_children_edges)
                new_node = prog.add_terminal_node(sym, None, value) if not parent_node_id == -1 else \
                    prog.add_terminal_node(sym, None, value, start_node=True)

                if not parent_node_id == -1:
                    self.set_parent_child(prog, parent_node_id, new_node.id)
                    # print("p children:", p.to_children_edges)
                # print("after prog:", prog)
                return_progs.append(prog)
        else:
            raise NotImplementedError

        return return_progs


class GuardBotUpSynthesizer(BottomUpSynthesizer):
    def __init__(self, grammar: GuardCFG, size_limit=50, random_sample=False):
        super(GuardBotUpSynthesizer, self).__init__(grammar, size_limit=size_limit, random_sample=random_sample)
        self.opt_f1 = 0.0
        self.worklist = PriorityQueue(get_priority_depth_reverse)
        # this stores all possible predicate (this can be precomputed)
        self.predicate_pool: List[Program] = self.instantiate_prog_with_sym(None, self.grammar.name_to_sym['np_2'], -1)
        self.guard_explored = 0

    def reinit(self):
        # self.worklist = PriorityQueue(priority=get_priority_f1_reverse)
        self.worklist = PriorityQueue(get_priority_depth_reverse)
        self.id_counter = itertools.count(start=1)  # TODO: not sure about this
        self.opt_f1 = 0.0

    def get_next_guard(self, task: Task, examples: PredExample, pred_enum_lim, pruning=True):

        self.worklist.put_all(self.init_prog(task))
        synthesized_guard_count = 0

        while not self.worklist.is_empty():
            curr_sec_locator: LocatorProgram = self.worklist.pop()

            # print("curr_sec_locator:", curr_sec_locator)
            if pruning:
                if curr_sec_locator.get_avg_f1() < self.opt_f1:
                    continue

            if curr_sec_locator.curr_sym.name.startswith("v"):
                true_only = len(examples.neg_benchmarks) == 0
                # true_only = False
                for psi in self.generate_guards(curr_sec_locator, true_only):

                    if psi.pred is not None:
                        self.guard_explored += 1
                        pred_in_guard = psi.pred.exec()
                        # some heuristic filter
                        if "hasStrEnt" in pred_in_guard and (
                                (("not" in pred_in_guard) and not ("const_str" in pred_in_guard and "DATE" in
                                                                   pred_in_guard)) or
                                ("const_str" in pred_in_guard and "LAW" in pred_in_guard) or
                                ("const_str" in pred_in_guard and "ORDINAL" in pred_in_guard) or 
                                ("const_str" in pred_in_guard and "NOUN" in pred_in_guard)
                        ):
                            continue

                    # print("psi:", psi)
                    pred_exec_context = self.evaluator.eval_prod_guard(task, examples, psi, pruning=pruning)
                    if not pred_exec_context.pruned and pred_exec_context.get_score() == 1.0:
                        yield psi
                    else:
                        pass
                        # print("pruned")

                    synthesized_guard_count += 1
                    if synthesized_guard_count > pred_enum_lim:
                        break

            for prod in self.infer_applicable_prods_bot_up(curr_sec_locator, curr_sec_locator.curr_sym,
                                                           curr_sec_locator.start_node.id):

                new_progs: List[Program] = self.expand_prog_with_production(curr_sec_locator,
                                                                            curr_sec_locator.curr_sym, prod)

                # print("new_progs:", new_progs)

                for prog in new_progs:

                    # print("prog:", prog.exec())
                    self.evaluator.eval_prod(task, examples.pos_benchmarks, prog, prod, pruning=pruning)

                    # prog_str = prog.exec()
                    # avg_recall = prog.get_avg_recall(approx=False)
                    # def append_section_matching_prod_name(prog_str):
                    #     if "matchKeyword" in prog_str:
                    #         best_section_matching_prod_name.append("matchKeyword")
                    #     elif "matchQA" in prog_str:
                    #         best_section_matching_prod_name.append("matchQA")
                    #     elif "matchSection1" in prog_str:
                    #         best_section_matching_prod_name.append("matchSection1")
                    #     elif "matchSection2" in prog_str:
                    #         best_section_matching_prod_name.append("matchSection2")
                    #
                    # if prod.return_symbol.name == "en":
                    #     if avg_recall > best_section_matching_recall:
                    #         best_section_matching_recall = avg_recall
                    #         best_section_matching_prod_name = []
                    #         append_section_matching_prod_name(prog_str)
                    #     elif avg_recall == best_section_matching_recall:
                    #         append_section_matching_prod_name(prog_str)

                    # print('prog.get_avg_f1():', prog.get_avg_f1())
                    if not pruning:
                        self.worklist.put(prog)
                    elif prog.get_avg_f1() >= self.opt_f1 and not prog.check_empty_f1(approx=False):
                        # print("avg stats: {}, {}, {}".format(prog.get_avg_f1(approx=False),
                        #                                      prog.get_avg_precision(approx=False),
                        #                                      prog.get_avg_recall(approx=False)))
                        # print("approx stats: {}, {}, {}".format(prog.get_avg_f1(),
                        #                                         prog.get_avg_precision(),
                        #                                         prog.get_avg_recall()))

                        # print("detailed approx stats: {}".format(prog.approx_stats))
                        self.worklist.put(prog)

        # print("here")
        return None

    def init_prog(self, task: Task, examples: List[Example] = None, pruning=False) -> List[IEProgram]:

        return_progs = []

        for sym in self.grammar.base_case_init_sym:
            progs = self.instantiate_prog_with_sym(None, sym, -1)
            for prog in progs:
                prog.stats = {'default': (1.0, 1.0, 1.0, 0.0)}
                return_progs.append(prog)
            # return_progs.extend(self.instantiate_prog_with_sym(None, sym, -1))

        return return_progs

    def generate_guards(self, locator_prog: LocatorProgram, true_only=False) -> List[GuardProgram]:

        if true_only:
            guard = GuardProgram(next(self.id_counter))
            guard.create_top_level_op(self.grammar.name_to_prod['AnySat'])
            guard.section_locator = locator_prog
            guard.pred = IEProgram(next(guard.id_counter), 'lambda x: dsl._true(x)', 'lambda: True')

            return [guard]

        ret_guards = []
        for prod in self.grammar.start_sym.prods:
            if prod.operator_name == "AnySat":
                for predicate in self.predicate_pool:
                    guard = GuardProgram(next(self.id_counter))
                    guard.create_top_level_op(prod)
                    guard.section_locator = locator_prog
                    guard.pred = predicate
                    ret_guards.append(guard)
            else:
                guard = GuardProgram(next(self.id_counter))
                guard.create_top_level_op(prod)
                guard.section_locator = locator_prog
                ret_guards.append(guard)
        return ret_guards


class ExtractorBotUpSynthesizer(BottomUpSynthesizer):
    def __init__(self, grammar: ExtractorCFG, size_limit=50, random_sample=False):
        super(ExtractorBotUpSynthesizer, self).__init__(grammar, size_limit=size_limit, random_sample=random_sample)

    def reinit(self):
        self.worklist = PriorityQueue(priority=get_priority_f1_reverse)
        self.id_counter = itertools.count(start=1)

    def synthesize(self, task: Task, examples: List[ExtractExample], enum_limit, depth_lim, pruning, optim_f1,
                   locator_prog_id=None, decomposition=True) \
            -> Tuple[List[SynthProgContextExt], float]:
        return super(ExtractorBotUpSynthesizer, self).synthesize(task, examples, enum_limit, depth_lim, pruning,
                                                                 optim_f1, locator_prog_id, decomposition)

    def synthesize_best(self, task: Task, examples: List[ExtractExample], enum_limit, depth_lim, pruning, optim_f1) \
            -> List[SynthProgContextExt]:
        optim_synthesized_prog = self.synthesize(task, examples, enum_limit, depth_lim, pruning, optim_f1)
        if len(optim_synthesized_prog) > 0:
            return Subset([e.name for e in examples], optim_synthesized_prog, optim_synthesized_prog[0].get_stats('f1'))
        else:
            print("no valid extractor")
            return []

    def init_prog(self, task: Task, examples: List[Example] = None, pruning=False) -> List[IEProgram]:
        # return_progs = []

        # in the new grammar, the only basecase is ExtractContent(x)
        progs = self.instantiate_prog_with_sym(None, self.grammar.name_to_sym['e'], -1,
                                               [self.grammar.name_to_prod['ExtractContent']])

        # for prog in progs:
        #     if examples is not None:
        #         self.evaluator.eval_prod(task, examples, prog, self.grammar.name_to_prod['ExtractContent'],
        #                                  context_format=True, pruning=pruning)
        #     prog.stats = {'default': (1.0, 1.0, 1.0, 0.0)}  # need to change this since this program is executable
        #     return_progs.append(prog)

        return progs

    def eval_prog(self, task: Task, examples: List[ExtractExample], curr_p: IEProgram):
        raise NotImplementedError
