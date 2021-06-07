import copy
import hashlib
import itertools
import random
import re

import numpy as np
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple
from lib.grammar.symbol import Symbol, TerminalSymbol, NonterminalSymbol
from lib.grammar.production import Production
from lib.interpreter.context import SynthProgContextExt

anysat_locator_pred_regex_pattern = re.compile(r'AnySat[(](.+),[ ]?(lambda.*)[)]')
issingleton_locator_pred_regex_pattern = re.compile(r'IsSingleton[(](.+)[)]')


class Node:
    def __init__(self, _id: int):
        self.id = _id
        # self.parent = parent
        self.is_hole = False
        self.sym = None

    def repr_helper(self, args):
        raise NotImplementedError

    def exec_helper(self, args):
        raise NotImplementedError


class NonterminalNode(Node):
    def __init__(self, _id: int, sym: NonterminalSymbol, prod: Production,
                 depth=-1, sym_depth=-1):
        super(NonterminalNode, self).__init__(_id)
        self.prod = prod
        self.sym = sym
        self.depth = depth
        self.sym_depth = sym_depth

    def repr_helper(self, args):
        return self.prod.repr_helper(args)

    def exec_helper(self, args, exclude_prefix):
        return self.prod.exec_helper(args, exclude_prefix)

    def __repr__(self):
        return "{}".format(self.prod.operator_name)


class VariableNode(Node):
    def __init__(self, _id: int, sym: Symbol):
        super(VariableNode, self).__init__(_id)
        self.sym = sym

    def repr_helper(self, args):
        return "?_{}[{}]".format(self.id, self.sym)

    def exec_helper(self, args):
        raise NotImplementedError


class TerminalNode(Node):
    def __init__(self, _id: int, sym: TerminalSymbol, value):
        super(TerminalNode, self).__init__(_id)
        self.sym = sym
        self.value = value

    def repr_helper(self, args):
        return str(self.value)

    def exec_helper(self, args):
        if self.sym.name == "label":
            return "\'{}\'".format(self.value)
        else:
            return str(self.value)

    def __repr__(self):
        return self.value if self.value is not None else self.sym.name


class Program:
    def __init__(self, _id, prog_str=None, prog_repr=None):
        self.id = _id
        self.prog_str = prog_str
        if prog_repr is not None:
            self.prog_repr = prog_repr
        elif prog_repr is None and prog_str is not None:
            self.prog_repr = prog_str
        else:
            self.prog_repr = prog_repr

        self.start_node: Node = None
        self.nodes: Dict[int, Node] = {}
        self.to_children_edges: Dict[int, List[int]] = {}
        self.to_parent_edges: Dict[int, int] = {}

        self.var_nodes: OrderedDict[int, str] = OrderedDict()
        self.selected_var_id = -1
        self.select_var_helper = 0

        self.cost = 0
        self.id_counter = itertools.count()
        self.depth = -1

        # the following is for bottom-up only
        self.curr_sym: Symbol = None
        self.exec_results: Dict[str, object] = {}
        self.approx_stats: Dict[str, Tuple[float, float, float, float]] = {}
        self.stats: Dict[str, Tuple[float, float, float, float]] = {}

    def check_empty_f1(self, approx=True):
        if approx:
            return any([s[2] == 0 for s in self.approx_stats.values()])
        else:
            return any([s[2] == 0 for s in self.stats.values()])

    def get_avg_f1(self, approx=True, given_set=None):
        if given_set is not None:
            if approx:
                return np.mean([x[1][2] for x in self.approx_stats.items() if x[1] in given_set])
            else:
                return np.mean([x[1][2] for x in self.stats.items() if x[1] in given_set])
        else:
            if approx:
                return np.mean([x[2] for x in self.approx_stats.values()])
            else:
                return np.mean([x[2] for x in self.stats.values()])

    def get_avg_precision(self, approx=True):
        if approx:
            return np.mean([x[0] for x in self.approx_stats.values()])
        else:
            return np.mean([x[0] for x in self.stats.values()])

    def get_avg_recall(self, approx=True):
        if approx:
            return np.mean([x[1] for x in self.approx_stats.values()])
        else:
            return np.mean([x[1] for x in self.stats.values()])

    def duplicate(self, _id):
        raise NotImplementedError

    def add_variable_node(self, sym: Symbol, parent: Node = None, sub_id=None):
        self.prog_repr = None
        if sub_id is None:
            new_node = VariableNode(next(self.id_counter), sym)
        else:
            new_node = VariableNode(sub_id, sym)

        self.nodes[new_node.id] = new_node

        if parent is not None:
            self.to_parent_edges[new_node.id] = parent.id

        self.var_nodes[new_node.id] = ""

        return new_node

    def add_terminal_node(self, sym: Symbol, parent: Node, value,
                          sub_id=None, start_node=False):
        assert isinstance(sym, TerminalSymbol)
        self.prog_repr = None

        if sub_id is None:
            new_node = TerminalNode(next(self.id_counter), sym, value)
        else:
            new_node = TerminalNode(sub_id, sym, value)

        if parent is not None:
            self.to_parent_edges[new_node.id] = parent.id
            assert isinstance(parent, NonterminalNode)
            self.depth = (parent.depth + 1) \
                if parent.depth >= self.depth else self.depth

        self.nodes[new_node.id] = new_node
        self.cost += new_node.sym.get_cost(value)

        return new_node

    def add_nonterminal_node(self, sym: Symbol, prod: Production, parent: Node,
                             children: List[int] = None,
                             sub_id=None, start_node=False, bot_up=False):
        assert isinstance(sym, NonterminalSymbol)
        self.prog_repr = None

        if sub_id is None:
            new_node = NonterminalNode(next(self.id_counter), sym, prod)
        else:
            new_node = NonterminalNode(sub_id, sym, prod)

        # TODO: all of these is probably not necessary (except the cost part)
        # if we are replacing a var node with a real node
        if parent is not None:
            assert isinstance(parent, NonterminalNode)
            self.to_parent_edges[new_node.id] = parent.id

        # problematic in bottomup context
        if start_node:
            self.start_node = new_node

        self.to_children_edges[new_node.id] = children
        self.nodes[new_node.id] = new_node
        self.cost += new_node.prod.cost

        # deal with depth issue here
        if not bot_up:
            if parent is not None:
                new_node.depth = (parent.depth + 1)
            if start_node:
                new_node.depth = 1
        else:
            self.depth = 1 if (start_node and self.depth == -1) else (self.depth + 1)

        return new_node

    def delete_var_node(self, d_id):
        self.prog_repr = None
        del self.var_nodes[d_id]

    def select_var_node(self):
        self.prog_repr = None
        # TODO: assumption:
        # Dict preserve insertion order (holds true for Python 3.6 >)
        selected_idx = 0

        if self.select_var_helper >= len(self.var_nodes):
            # reset counter
            self.select_var_helper = 0
        else:
            selected_idx = self.select_var_helper
            self.select_var_helper += 1

        self.selected_var_id = list(self.var_nodes.items())[selected_idx][0]
        return self.nodes[self.selected_var_id]

    def set_children(self, parent: Node, children: List[Node]):
        self.prog_repr = None
        self.to_children_edges[parent.id] = [c.id for c in children]

    def get_children(self, parent: Node):
        return self.to_children_edges[parent.id]

    def set_parent(self, parent: Node, child: Node):
        assert isinstance(parent, NonterminalNode)
        self.prog_repr = None
        self.to_parent_edges[child.id] = parent.id

        if isinstance(child, NonterminalNode):
            child.depth = parent.depth + 1
        else:
            self.depth = (parent.depth + 1) \
                if parent.depth >= self.depth else self.depth

    def get_parent(self, child: Node):
        # print(self.start_node.id)
        if self.start_node.id == child.id:
            return None
        return self.nodes[self.to_parent_edges[child.id]]

    def get_node(self, _id: int):
        return self.nodes[_id]

    def is_concrete(self) -> bool:
        return len(self.var_nodes) == 0

    def repr_helper(self, node):
        if isinstance(node, TerminalNode) or isinstance(node, VariableNode):
            return node.repr_helper(None)

        assert isinstance(node, NonterminalNode)

        child_str = [
            self.repr_helper(self.nodes[child_node])
            for child_node in self.to_children_edges[node.id]]
        return node.repr_helper(child_str)

    def __repr__(self):
        """
        Return a string representation of the program.
        Use already computed string if available (to avoid excess computation).
        """
        if self.prog_repr is None:
            # self.prog_repr = "{}".format(self.repr_helper(self.start_node))
            self.prog_repr = "{}".format(self.exec_helper(self.start_node))
        # return self.prog_repr + "," + str(self.depth)
        return self.prog_repr

    def exec_helper(self, node, exclude_prefix: bool = False):
        if isinstance(node, TerminalNode):
            return node.exec_helper(None)

        assert isinstance(node, NonterminalNode)

        child_func = [
            self.exec_helper((self.nodes[child_node]), exclude_prefix)
            for child_node in self.to_children_edges[node.id]]
        return node.exec_helper(child_func, exclude_prefix)

    def exec(self):
        if self.prog_str is not None:
            if not self.prog_str.startswith('lambda'):
                return 'lambda:' + self.prog_str
            else:
                return self.prog_str

        # print(self.start_node.sym.name)
        if self.start_node.sym.name == "np_2":
            return self.exec_helper(self.start_node)

        return "lambda: {}".format(self.exec_helper(self.start_node))

    def get_id(self):
        """
        Return a unique id representing this program string.
        Currently using MD5 hash assuming no collisions.
        """
        repr_str = str(self)
        prog_hash = hashlib.md5(repr_str.encode('utf-8')).hexdigest()
        return prog_hash


class IEProgram(Program):
    def __init__(self, _id, prog_str=None, prog_repr=None):
        super(IEProgram, self).__init__(_id, prog_str, prog_repr)

    def duplicate(self, _id):
        ret = IEProgram(_id, None, None)
        ret.start_node = self.start_node
        ret.nodes = self.nodes.copy()
        ret.var_nodes = self.var_nodes.copy()
        ret.to_children_edges = self.to_children_edges.copy()
        ret.to_parent_edges = self.to_parent_edges.copy()
        ret.selected_var_id = self.selected_var_id
        ret.select_var_helper = self.select_var_helper
        ret.cost = self.cost
        ret.curr_sym = self.curr_sym
        ret.depth = self.depth
        # Do this because tee returns a tuple
        ret.id_counter = itertools.tee(self.id_counter)[1]

        ret.exec_results = self.exec_results

        return ret

    def get_id(self):
        return "p" + super().get_id()


# need to break down into top_level_node, section_locator
class LocatorProgram(IEProgram):
    def __init__(self, _id, prog_str=None, prog_repr=None):
        super(LocatorProgram, self).__init__(_id, prog_str, prog_repr)

    def duplicate(self, _id):
        ret = LocatorProgram(_id)
        ret.start_node = self.start_node
        ret.nodes = self.nodes.copy()
        ret.var_nodes = self.var_nodes.copy()
        ret.to_children_edges = self.to_children_edges.copy()
        ret.to_parent_edges = self.to_parent_edges.copy()
        ret.selected_var_id = self.selected_var_id
        ret.select_var_helper = self.select_var_helper
        ret.cost = self.cost
        # Do this because tee returns a tuple
        ret.id_counter = itertools.tee(self.id_counter)[1]

        return ret

    def get_id(self):
        return "l" + super().get_id()


class GuardProgram(Program):
    def __init__(self, _id, prog_str=None, prog_repr=None):
        super(GuardProgram, self).__init__(_id, prog_str, prog_repr)
        self.section_locator: LocatorProgram = None
        self.top_level_op: Node = None
        self.pred: Program = None

    def create_top_level_op(self, prod: Production):
        self.top_level_op = NonterminalNode(next(self.id_counter), prod.return_symbol, prod)

    def get_id(self):
        return "g" + super().get_id()

    def __repr__(self):
        if self.pred is not None:
            return "{}({},{})".format(self.top_level_op, self.section_locator, self.pred)
        else:
            return "{}({})".format(self.top_level_op, self.section_locator)

    def exec(self):
        if self.pred is not None:
            return "lambda: {}({},{})".format(self.top_level_op, self.section_locator, self.pred)
        else:
            return "lambda: {}({})".format(self.top_level_op, self.section_locator)


class BranchProgram(Program):
    def __init__(self, _id: int, guards: List[GuardProgram], extractors_contexts: List[SynthProgContextExt],
                 extractors: List[IEProgram] = None):
        super(BranchProgram, self).__init__(_id)
        self.guards: List[GuardProgram] = guards
        self.extractors = extractors
        self.extractor_contexts: List[SynthProgContextExt] = extractors_contexts  # technically this shouldn't be
        # here...

    def __repr__(self):
        if self.extractors is None:
            return 'bp{}:({{{}}},{{{}}})'.format(
                self.id, [pred.get_id() for pred in self.guards],
                [prog_context.program.get_id() for prog_context in self.extractor_contexts])
        else:
            return 'bp{}:({{{}}},{{{}}})'.format(
                self.id, [pred.get_id() for pred in self.guards],
                [prog_context.get_id() for prog_context in self.extractors])

    def print_prog(self):
        if self.extractors is None:
            return 'bp{}: {{{}}} -> {{{}}}'.format(self.id, [pred.exec() for pred in self.guards],
                                                   [prog_context.program.exec() for prog_context in
                                                    self.extractor_contexts])
        else:
            return 'bp{}: {{{}}} -> {{{}}}'.format(self.id, [pred.exec() for pred in self.guards],
                                                   [prog_context.exec() for prog_context in
                                                    self.extractors])

    def print_result(self):
        return 'bp{}:'.format(self.id) + str({"output": (str(self.extractor_contexts[0].benchmark_results)), "(p, r, "
                                                                                                             "f1)": (
            self.extractor_contexts[
                0].get_stats('precision'),
            self.extractor_contexts[0].get_stats(
                'recall'),
            self.extractor_contexts[0].get_stats(
                'f1')), "taskf1": self.extractor_contexts[0].task_f1})

    def get_pred_prog_list(self, top_program_id, branch_id):
        if self.extractors is None:
            return [{'P_id': top_program_id, 'branch': branch_id, 'bp_id': self.id, 'id': guard.get_id(),
                     'program_type': 'g', 'program':
                         guard.exec()} for guard in self.guards] + \
                   [{'P_id': top_program_id, 'branch': branch_id, 'bp_id': self.id, 'id': extractor.program.get_id(),
                     'program_type': 'e', 'program': extractor.program.exec()} for extractor in self.extractor_contexts]
        else:
            return [{'P_id': top_program_id, 'branch': branch_id, 'bp_id': self.id, 'id': guard.get_id(),
                     'program_type': 'g', 'program': guard.exec()} for guard in self.guards] + \
                   [{'P_id': top_program_id, 'branch': branch_id, 'bp_id': self.id, 'id': extractor.get_id(),
                     'program_type': 'e', 'program': extractor.exec()} for extractor in self.extractors]


class BranchNode(Node):
    def __init__(self, _id: int, progs=None):
        super(BranchNode, self).__init__(_id)
        if progs is not None:
            self.progs = progs
        else:
            self.progs: List[BranchProgram] = []
        self.failed = False
        self.f1 = 0

        self.flatten_progs = None  # for sampling purpose

    def sample_random_prog(self, sample=True) -> BranchProgram:
        if self.flatten_progs is None:
            self.flatten_progs = []
            for prog in self.progs:
                assert len(prog.guards) == 1
                if prog.extractors is not None:
                    for extractor in prog.extractors:
                        self.flatten_progs.append((prog.guards[0], extractor))
                else:
                    for extractor_context in prog.extractor_contexts:
                        self.flatten_progs.append((prog.guards[0], extractor_context.program))

        if sample:
            sampled_pair = self.flatten_progs[random.randint(0, len(self.flatten_progs) - 1)]
            return BranchProgram(0, [sampled_pair[0]], None, extractors=[sampled_pair[1]])

    def add_branch_prog(self, prog: BranchProgram):
        self.progs.append(prog)

    def repr_helper(self, args):
        raise NotImplementedError

    def exec_helper(self, args):
        raise NotImplementedError

    def __repr__(self):
        return 'b{}: {{{}}}'.format(
            self.id, [prog.get_id() for prog in self.progs])

    def print_progs(self):
        return 'b{}: {{{}}}'.format(self.id, [prog.print_prog() for prog in self.progs])

    def print_results(self):
        return "b{}: {{{}}}".format(self.id, [prog.print_result() for prog in self.progs])

    def get_pred_prog_list(self, top_program_id):
        output_list = []
        for prog in self.progs:
            output_list.extend(prog.get_pred_prog_list(top_program_id, self.id))
        return output_list


class TopLevelProgram:
    def __init__(self, _id=1, init=None, force_id=False):
        self.id = _id
        self.force_id = force_id
        self.nodes: Dict[int, BranchNode] = {}
        self.exec_order = []
        self.id_counter = itertools.count()
        self.depth = 0

        if init is not None:
            self.init_prog(init)

    def mk_node(self, progs: List[BranchProgram]) -> Tuple[int, BranchNode]:
        self.prog_repr = None
        new_node_id = next(self.id_counter)
        new_node = BranchNode(new_node_id, progs)
        self.nodes[new_node_id] = new_node
        self.exec_order.append(new_node_id)
        self.depth += 1

        return new_node_id, new_node

    def init_prog(self, guard_extractor_pairs: List[Tuple]):
        for id, (guard, extractor) in enumerate(guard_extractor_pairs):
            if isinstance(guard, str) or isinstance(extractor, str):
                self.force_id = True
            if isinstance(guard, str):
                # need to find out the locators and the preds
                # print("guard:", guard)
                guard_prog = GuardProgram(0, prog_str=guard)
                if "AnySat" in guard:
                    locator_pred_candidates = re.findall(anysat_locator_pred_regex_pattern, guard)[0]
                    # print("locator_pred_candidates:", locator_pred_candidates)
                    assert len(locator_pred_candidates) == 2
                    locator = locator_pred_candidates[0]
                    pred = locator_pred_candidates[1]
                    guard_prog.section_locator = LocatorProgram(0, prog_str=locator)
                    guard_prog.pred = Program(0, prog_str=pred)
                    guard_prog.top_level_op = "AnySat"
                else:
                    assert "IsSingleton" in guard
                    locator_candidates = re.findall(issingleton_locator_pred_regex_pattern, guard)[0]
                    assert len(locator_candidates) == 1
                    locator = locator_candidates[0]
                    guard_prog.section_locator = LocatorProgram(0, prog_str=locator)
                    guard_prog.top_level_op = "IsSingleton"

                guard = [guard_prog]
            if isinstance(extractor, str):
                extractor = [IEProgram(id, prog_str=extractor)]
            branch_prog = BranchProgram(id, guard, None, extractors=extractor)
            self.mk_node([branch_prog])

    def get_branch(self, _id) -> Tuple[int, BranchNode]:
        node = self.nodes[_id]
        return _id, node

    def get_id(self):
        if self.force_id:
            return self.id
        return "P" + hashlib.md5(str(self).encode('utf-8')).hexdigest()

    def exec(self):
        return "|".join([self.nodes[i].print_progs() for i in self.exec_order])

    def __repr__(self):
        return "|".join([str(self.nodes[i]) for i in self.exec_order])

    def get_pred_prog_list(self):
        output_list = []
        for i in self.exec_order:
            output_list.extend(self.nodes[i].get_pred_prog_list(self.get_id()))
        return output_list

    # NOTE: for each branch we only have one program
    def get_avg_prog_size(self):
        prog_size = 0
        for branch in self.nodes.values():
            prog_size += len(branch.progs[0].guards[0].nodes)
            if "matchQA" in branch.progs[0].guards[0].exec():
                prog_size += 1
            if branch.progs[0].extractors is not None:
                prog_size += len(branch.progs[0].extractors[0].nodes)
            else:
                prog_size += len(branch.progs[0].extractor_contexts[0].program.nodes)
        return prog_size
