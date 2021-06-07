import random
from collections import defaultdict
from typing import List

from lib.config import PRINT_EVAL_DETAILS
from lib.grammar.production import Production
from lib.interpreter.context import ResultContext, SynthProgContextGuard, SynthProgContextTop, SynthProgContextExt, \
    NodeContext, StrContext
from lib.interpreter.executor import Executor
from lib.program import TopLevelProgram, IEProgram, GuardProgram, NonterminalNode
from lib.spec import ExtractExample, PredExample, Task
from lib.utils.misc_utils import printc
from lib.utils.result_utils import compute_f1


class Evaluator:
    def __init__(self, executor=Executor()):
        self.executor = executor

        # assumption: the examples to be executed should be the same (only used in train time)
        # List[(output, tuple, tuple)] for bottom-up eval-prod
        # context for top=down
        self.locator_extractor_cache = defaultdict(dict)
        self.pred_cache = {}

        self.locator_extractor_cache_hit = 0
        self.locator_extractor_cache_miss = 0
        self.pred_cache_hit = 0
        self.pred_cache_miss = 0
        self.entity_pass_hit = 0

    # for evaluate locator and extractor
    def eval_prod(self, task: Task, examples: List[ExtractExample], program: IEProgram, prod: Production,
                  context_format=False, pruning=True, locator_prog_id=None):

        printc(PRINT_EVAL_DETAILS, "in eval prod ", program.exec(), " ")
        if locator_prog_id is None:
            prog_id = program.get_id()
        else:
            prog_id = (locator_prog_id, program.get_id())

        # if self.locator_extractor_cache.get(prog_id) is None:
        #     self.locator_extractor_cache[prog_id] = {}

        program_so_far_result = program.exec_results
        exec_results = {}
        approx_stats = {}
        stats = {}

        f1s = []
        precisions = []
        recalls = []

        for example in examples:

            result_stat = self.locator_extractor_cache[prog_id].get(example.name)
            # print("result_stat:", result_stat)
            if result_stat is None:
                self.locator_extractor_cache_miss += 1
                result = self.executor.exec_prod(program, prod, example, task, program_so_far_result, pruning=pruning)
                # print("result: ", result)

                if pruning and isinstance(result, str):
                    assert result == "ENTITY_PASS"
                    result = []
                    stat = (1.0, 0.0, 0.0, 0.0)
                    approx_stat = (1.0, 0.0, 0.0, 0.0)
                    self.entity_pass_hit += 1
                    # print("entity pass")
                elif prod.return_symbol.name == "n" or prod.return_symbol.name == "n_1" \
                        or prod.return_symbol.name == "v":
                    tmp_str_context = []
                    for elem in result:
                        p_n = elem.tree_node
                        l_n = elem.list_node
                        bindex = (p_n.id, l_n.id if l_n is not None else 0)

                        tmp_str_context.append(StrContext(bindex, self.executor.dsl.string_context_helper(elem,
                                                                                                          example.name,
                                                                                                          bindex[0],
                                                                                                          bindex[1]),
                                                          None))
                    # print("tmp_str_context:", tmp_str_context)
                    stat = compute_f1(str(tmp_str_context), str(example.gt.gt_str))
                    approx_stat = (1.0, stat[1], ((2 * stat[1]) / (1 + stat[1])), 0.0)
                    # print("approx_stat:", approx_stat)
                else:
                    stat = compute_f1(str(result), str(example.gt.gt_str))
                    approx_stat = (1.0, stat[1], ((2 * stat[1]) / (1 + stat[1])), 0.0)
                self.locator_extractor_cache[prog_id][example.name] = (result, stat, approx_stat)
            else:
                self.locator_extractor_cache_hit += 1
                result, stat, approx_stat = result_stat

            exec_results[example.name] = result
            stats[example.name] = stat
            approx_stats[example.name] = approx_stat

            precisions.append(stat[0])
            recalls.append(stat[1])
            f1s.append(stat[2])

        program.exec_results = exec_results
        program.approx_stats = approx_stats
        program.stats = stats

        # print("exec_results {}: {}".format(program.get_id(), exec_results))

        if context_format:
            return SynthProgContextExt(program, dict([(example.name, ResultContext(
                exec_results[example.name],
                stats[example.name][0],
                stats[example.name][1],
                stats[example.name][2],
                stats[example.name][3],
                example.name,
                extractor_id=prog_id)) for example in examples]),
                                       f1s, precisions, recalls)

            # print("result:", result)

    # for evaluate extractor and locator
    def eval_extractor(self, task: Task, examples: List[ExtractExample],
                       program: IEProgram) -> SynthProgContextExt:

        assert not isinstance(program, TopLevelProgram)

        prog_id = program.get_id()

        if self.locator_extractor_cache.get(prog_id) is None:
            self.locator_extractor_cache[prog_id] = {}

        benchmark_results = {}
        f1_all = []
        precision_all = []
        recall_all = []
        for example in examples:
            result = self.locator_extractor_cache[prog_id].get(example.name)
            if result is None:
                result = self.executor.exec_prog(program, example, task)
                assert isinstance(program.start_node, NonterminalNode)
                if program.start_node.prod.return_symbol.name == "n" or program.start_node.prod.return_symbol.name == "n_1" \
                        or program.start_node.prod.return_symbol.name == "v":
                    tmp_str_context = []
                    for elem in result.output:
                        p_n = elem.tree_node
                        l_n = elem.list_node
                        bindex = (p_n.id, l_n.id if l_n is not None else 0)

                        tmp_str_context.append(StrContext(bindex, self.executor.dsl.string_context_helper(elem,
                                                                                                          example.name,
                                                                                                          bindex[0],
                                                                                                          bindex[1]),
                                                          None))
                    # print("tmp_str_context:", tmp_str_context)
                    stat = compute_f1(str(tmp_str_context), str(example.gt.gt_str))
                    (precision, recall, f1score, na) = stat
                else:
                    (precision, recall, f1score, na) = compute_f1(str(result.output), str(example.gt.gt_str))

                result.update_statistics(precision, recall, f1score, na)
                approx_stat = (1.0, precision, ((2 * recall) / (1 + recall)), 0.0)
                self.locator_extractor_cache[prog_id][example.name] = result, (
                    precision, recall, f1score, na), approx_stat
            # result = ResultContext("")

            benchmark_results[example.name] = result
            f1_all.append(result.f1score)
            precision_all.append(result.precision)
            recall_all.append(result.recall)

        printc(PRINT_EVAL_DETAILS, benchmark_results)
        printc(PRINT_EVAL_DETAILS, f1_all, precision_all, recall_all)

        return SynthProgContextExt(program, benchmark_results, f1_all, precision_all, recall_all)

    # for evaluate guard with intermediate resutls evaluated
    def eval_prod_guard(self, task: Task, example: PredExample, program: GuardProgram, pruning=True) -> \
            SynthProgContextGuard:

        # print("program:", program.exec())
        pred_id = program.get_id()

        if self.pred_cache.get(pred_id) is None:
            self.pred_cache[pred_id] = {}

        positive_benchmarks_results = []
        pos_total = 0
        pos_accept = 0
        # print("pos examples")
        for b in example.pos_benchmarks:
            pos_total += 1

            exec_res = self.pred_cache[pred_id].get(b.name)
            # print("pred:", program)
            if exec_res is None:
                self.pred_cache_miss += 1
                # exec_res = self.executor.exec_prog_guard(program, b, task)
                if pruning:
                    exec_res = self.executor.exec_prod_guard(program, b, task, pruning=pruning,
                                                             locator_exec_res=program.section_locator.exec_results[
                                                                 b.name])
                else:
                    locator_exec_result = self.executor.exec_prog(program.section_locator, b, task).output
                    exec_res = self.executor.exec_prod_guard(program, b, task, pruning=pruning,
                                                 locator_exec_res=locator_exec_result)
                self.pred_cache[pred_id][b.name] = exec_res
            else:
                self.pred_cache_hit += 1

            # print("{} exec_res {}:".format(b.name, exec_res))

            positive_benchmarks_results.append((b.name, exec_res))
            if exec_res.output.output:
                pos_accept += 1
            else:
                return SynthProgContextGuard(program, {}, pruned=True)

        negative_benchmarks_results = []
        neg_total = 0
        neg_reject = 0
        # print("neg examples")

        for b in example.neg_benchmarks:
            neg_total += 1
            exec_res = self.pred_cache[pred_id].get(b.name)

            if exec_res is None:
                self.pred_cache_miss += 1
                exec_res = self.guard_exec_helper(program, b, task)
                self.pred_cache[pred_id][b.name] = exec_res
            else:
                self.pred_cache_hit += 1
            # print(exec_res)
            # print("{} exec_res {}:".format(b.name, exec_res))
            negative_benchmarks_results.append((b.name, exec_res))
            if not exec_res.output.output:
                neg_reject += 1
            else:
                SynthProgContextGuard(program, {}, pruned=True)

        return SynthProgContextGuard(program, dict(positive_benchmarks_results + negative_benchmarks_results),
                                     pos_total=pos_total, pos_accept=pos_accept,
                                     neg_total=neg_total, neg_reject=neg_reject)

    def guard_exec_helper(self, guard: GuardProgram, b: ExtractExample, task: Task, calc_stats=True):
        # NOTE: since section locators are never executed on neg benchmarks, we need to execute it here
        locator_prog_id = guard.section_locator.get_id()
        locator_exec_result = self.locator_extractor_cache[locator_prog_id].get(b.name)
        if locator_exec_result is None:
            locator_exec_result = self.executor.exec_prog(guard.section_locator, b, task)
            if calc_stats:
                assert isinstance(guard.section_locator.start_node, NonterminalNode)
                if guard.section_locator.start_node.prod.return_symbol.name == "n" or \
                        guard.section_locator.start_node.prod.return_symbol.name == "n_1" or \
                        guard.section_locator.start_node.prod.return_symbol.name == "v":
                    tmp_str_context = []
                    for elem in locator_exec_result.output:
                        p_n = elem.tree_node
                        l_n = elem.list_node
                        bindex = (p_n.id, l_n.id if l_n is not None else 0)

                        tmp_str_context.append(StrContext(bindex, self.executor.dsl.string_context_helper(elem,
                                                                                                        b.name,
                                                                                                        bindex[0],
                                                                                                        bindex[1]),
                                                        None))
                    # print("tmp_str_context:", tmp_str_context)
                    stat = compute_f1(str(tmp_str_context), str(b.gt.gt_str))
                else:
                    stat = compute_f1(str(locator_exec_result), str(b.gt.gt_str))
            else:
                stat = (0, 0, 0, 0)

            approx_stat = (1.0, stat[1], ((2 * stat[1]) / (1 + stat[1])), 0.0)
            self.locator_extractor_cache[locator_prog_id][b.name] = locator_exec_result.output, \
                                                                    stat, approx_stat
            locator_exec_result = locator_exec_result.output
        else:
            locator_exec_result = locator_exec_result[0]

        exec_res = self.executor.exec_prod_guard(guard, b, task, pruning=False,
                                                 locator_exec_res=locator_exec_result)
        return exec_res

    # for evaluate guard from start
    def eval_guard(self, task: Task, example: PredExample, program: GuardProgram) -> \
            SynthProgContextGuard:
        return self.eval_prod_guard(task, example, program, pruning=False)

    def eval_top_level_helper(self, top_level_prog: TopLevelProgram, benchmarks: List[ExtractExample], task: Task):

        printc(PRINT_EVAL_DETAILS, "in eval_top_level_helper:", top_level_prog.exec(), ", ", top_level_prog)

        results = {}
        curr_benchmarks = benchmarks

        for branch_id in top_level_prog.exec_order:

            _, branch_node = top_level_prog.get_branch(branch_id)
            assert branch_node.progs[0].extractors is not None
            guard = branch_node.progs[0].guards[0]
            extractor = branch_node.progs[0].extractors[0]
            next_benchmarks = []
            for b in curr_benchmarks:
                print(b.name)
                pred_res = self.guard_exec_helper(guard, b, task, calc_stats=False)
                # print('pred_res:', pred_res)
                if pred_res.output.output:
                    section_locator_res = self.locator_extractor_cache[guard.section_locator.get_id()].get(b.name)
                    assert section_locator_res is not None
                    prog_res = self.executor.exec_prog(extractor, b, task, x=section_locator_res[0])
                    results[b.name] = ResultContext(prog_res.output, file_name=b.name, branch_id=branch_id,
                                                    extractor_id=extractor.get_id(),
                                                    guard_id=guard.get_id(),
                                                    pred_time=pred_res.pred_time,
                                                    prog_time=prog_res.prog_time)
                else:
                    next_benchmarks.append(b)
            curr_benchmarks = next_benchmarks

            # see there is still leftover, return []
            print("return empty benchmarks")
            for b in curr_benchmarks:
                print(b.name)
                results[b.name] = ResultContext([], file_name=b.name, branch_id=branch_id,
                                                extractor_id=extractor.get_id(),
                                                guard_id=guard.get_id(),
                                                pred_time=0,
                                                prog_time=0)

        return results

    # for evaluate TopLevelProgram
    def eval_toplevel(self, task: Task, benchmarks: List[ExtractExample], program: TopLevelProgram,
                      idx='1') -> SynthProgContextTop:

        assert not isinstance(program, IEProgram) or not isinstance(program, GuardProgram)

        # create a TopLevelProgeram
        exec_top_level_prog = TopLevelProgram('{}_{}'.format(str(program.get_id()), idx), force_id=True)
        for i in program.exec_order:
            _, branch_node = program.get_branch(i)
            exec_top_level_prog.mk_node(
                [branch_node.sample_random_prog()])  # need to sample one program from this branch_node

        benchmark_results = self.eval_top_level_helper(exec_top_level_prog, benchmarks, task)
        f1_all = []
        precision_all = []
        recall_all = []
        for b in benchmarks:
            (precision, recall, f1score, na) = compute_f1(repr(benchmark_results[b.name].output), str(b.gt.gt_str))
            benchmark_results[b.name].update_statistics(precision, recall, f1score, na)
            f1_all.append(f1score)
            precision_all.append(precision)
            recall_all.append(recall)
        return SynthProgContextTop(exec_top_level_prog, benchmark_results, f1_all, precision_all, recall_all)
