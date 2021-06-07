import time

from .context import ResultContext, PredContext
from .dsl import DSL
from ..grammar import constant
from ..grammar.production import Production
from ..grammar.symbol import NonterminalSymbol, TerminalSymbol
from ..program import TopLevelProgram, Program, GuardProgram, TerminalNode, NonterminalNode
from ..spec import Task, ExtractExample


class Executor:
    def __init__(self, dsl=None, preddsl=None):
        self.dsl = dsl if dsl is not None else DSL()

    def exec_prod(self, prog: Program, prod: Production, b: ExtractExample, task: Task, program_prev_result,
                  pruning=True):

        dsl = self.dsl
        self.dsl.pt = b.pt
        self.dsl.task = task
        const_str = task.const_str

        if prod.getName() == "ExtractContent":
            assert b.section_locator_exec_res is not None
            return self.dsl.ExtractContent(b.section_locator_exec_res)

        self.dsl.set_execute_prog_str(prog.exec(), task)

        processed_args = []
        direct_children_nodes = prog.get_children(prog.start_node)

        contain_ents = []

        for i in range(len(prod.argument_symbols)):

            if i == 0 and program_prev_result.get(b.name) is not None and not self.dsl.filter_const_getcsv_prog:
                processed_args.append(program_prev_result[b.name])
            else:
                curr_child_node = prog.get_node(direct_children_nodes[i])
                if isinstance(curr_child_node.sym, NonterminalSymbol):
                    prog_str = prog.exec_helper(curr_child_node, exclude_prefix=False)
                    if curr_child_node.prod.operator_name == "hasEntity":
                        child_node = prog.get_node(prog.get_children(curr_child_node)[0])
                        assert isinstance(child_node, TerminalNode)
                        contain_ents.append(child_node.value)
                    # print("prog_str:", prog_str)
                    if "ExtractContent" in prog_str:
                        x = b.section_locator_exec_res
                    if "GetRoot" in prog_str:
                        w = None
                    processed_args.append(eval(prog_str, locals()))
                else:
                    assert isinstance(curr_child_node.sym, TerminalSymbol)
                    if curr_child_node.sym.name == "w":
                        processed_args.append(None)
                    elif curr_child_node.sym.name == "const_str":
                        processed_args.append(task.const_str)
                    elif curr_child_node.sym.name == "q":
                        processed_args.append(task.q)
                    elif curr_child_node.sym.values is not None:
                        if curr_child_node.sym.name == "label":
                            contain_ents.append(curr_child_node.value)
                        processed_args.append(curr_child_node.value)

        # print("processed_args:", processed_args)

        # TODO: if the entity tag doesn't show up in the filtered entity, then directing filter it
        if pruning and len(contain_ents) > 0:
            # print(contain_ents)
            # print(constant.PRED_ENTITY)
            if contain_ents[0] not in constant.PRED_ENTITY:
                self.dsl.unset_execute_prog_str()
                return "ENTITY_PASS"

        if prod.getName() == "toV":
            prod_method = getattr(self.dsl, "GetRoot")
        else:
            prod_method = getattr(self.dsl, prod.getName())

        res = prod_method(*processed_args)

        self.dsl.unset_execute_prog_str()

        return res

    def exec_prod_guard(self, prog: GuardProgram, b: ExtractExample, task: Task,
                        pruning=True, locator_exec_res=None) -> ResultContext:
        dsl = self.dsl
        self.dsl.pt = b.pt
        self.dsl.task = task
        const_str = task.const_str

        self.dsl.set_execute_prog_str(prog.exec(), task)

        exec_start = time.perf_counter()
        locator_exec_res = locator_exec_res if locator_exec_res is not None else prog.section_locator.exec_results[
            b.name]
        # print("locator_exec_res:", locator_exec_res)
        top_level_node_opname = prog.top_level_op.prod.operator_name if isinstance(prog.top_level_op, NonterminalNode) \
            else prog.top_level_op
        # print("prog.top_level_op.prod:", prog.top_level_op.prod)
        if top_level_node_opname == "AnySat":
            # compile pred
            pred_str = prog.pred.exec()
            # print("pred_str:", pred_str)
            if pruning and "Ent" in pred_str and not any(ent in pred_str for ent in constant.PRED_ENTITY):
                o = PredContext(False, [n.match_section_nid for n in locator_exec_res])
            else:
                pred = eval(pred_str, locals())
                if "matchSection" in repr(prog):
                    o = dsl.AnySat(locator_exec_res, pred, matchSection=True)
                else:    
                    o = dsl.AnySat(locator_exec_res, pred)
        elif top_level_node_opname == "IsSingleton":
            o = dsl.isSingleton(locator_exec_res)
        else:
            raise NotImplementedError
        pred_finish = time.perf_counter()

        self.dsl.unset_execute_prog_str()

        return ResultContext(o, guard_id=prog.get_id(), file_name=b.name, pred_time=str(pred_finish - exec_start))

    def exec_prog(self, prog: Program, b: ExtractExample, task: Task, x=None, w=None) -> ResultContext:

        dsl = self.dsl
        prog_str = prog.exec()
        const_str = task.const_str
        q = task.q
        keyword = task.keyword

        self.dsl.set_execute_prog_str(prog_str, task)

        # print("prog_str:", prog_str)

        if isinstance(prog_str, list):
            raise NotImplementedError

        dsl.pt = b.pt
        dsl.task = task

        exec_start = time.perf_counter()
        output = eval(prog_str, locals())()
        prog_finish = time.perf_counter()

        self.dsl.unset_execute_prog_str()

        return ResultContext(output, extractor_id=prog.get_id(), file_name=b.name,
                             prog_time=str(prog_finish - exec_start))

    def exec_prog_guard(self, prog: GuardProgram, b: ExtractExample, task: Task) -> ResultContext:

        dsl = self.dsl
        prog_str = prog.exec()
        const_str = task.const_str
        q = task.q
        keyword = task.keyword
        if isinstance(prog_str, list):
            raise NotImplementedError

        dsl.pt = b.pt
        dsl.task = task

        # print("prog_str:", prog_str)

        exec_start = time.perf_counter()
        o = eval(prog_str, locals())()
        pred_finish = time.perf_counter()

        return ResultContext(o, guard_id=prog.get_id(), file_name=b.name, pred_time=str(pred_finish - exec_start))