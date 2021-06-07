import collections
import itertools
import time
from typing import List, Dict, Tuple

from lib.interpreter.context import SynthProgContextGuard, SynthProgContextExt, SynthProgContextTop, Subset
from lib.grammar.cfg import ExtractorCFG, GuardCFG
from lib.program import TopLevelProgram, BranchNode, BranchProgram, GuardProgram
from lib.spec import ExtractExample, Task, PredExample
from lib.synthesizer import ExtractorBotUpSynthesizer, GuardBotUpSynthesizer
from lib.utils.misc_utils import partition, get_priority_f1


class TopLevelSynthesizer:
    def __init__(self):
        self.benchmark_programs: Dict[str, List] = {}
        # self.extract_synth = ExtractorTopDownSynthesizer(ExtractorCFG())
        self.extract_synth = ExtractorBotUpSynthesizer(ExtractorCFG())
        # self.pred_synth = PredicateTopDownSynthesizer(PredicateCFG())
        self.guard_synth = GuardBotUpSynthesizer(GuardCFG())
        self.id_counter = itertools.count(start=1)

        self.branch_synthesized_cache = collections.defaultdict(dict)  # {example_list: {locator_id: ([List of
        # synthesized extractors], optim_f1)}}
        self.branch_synthesized_cache_hit = 0
        self.branch_synthesized_cache_miss = 0

    def synthesize(self, task: Task, benchmarks: List[ExtractExample], extractor_enum_lim=10, pred_enum_lim=10,
                   extractor_depth_lim=7, pruning=True, decomposition=True, example_partition=None) -> List[
        SynthProgContextTop]:
        print(benchmarks)
        best_top_level_prog_contexts: List[SynthProgContextTop] = []
        opt_score = 0.0
        all_example_partition = partition(benchmarks) if example_partition is None else example_partition

        for bp in all_example_partition:
            print("considering partition {}".format(str([[e.name for e in p_i] for p_i in bp])))
            prog_context = SynthProgContextTop(TopLevelProgram(next(self.id_counter)), {}, [], [], [])

            success = True
            for p_i in range(len(bp)):
                print("considering branch {}".format(str([b.name for b in bp[p_i]])))
                branch_id, branch_node = prog_context.program.mk_node([])
                self.synthesize_branch(branch_node, task, PredExample(bp[p_i], [e_i for rest_p_i in bp[(p_i + 1):]
                                                                                for e_i in rest_p_i]),
                                       extractor_enum_lim, pred_enum_lim, extractor_depth_lim, pruning, decomposition)
                if branch_node.failed:
                    print("imperfect classification")
                    success = False
                    break

                prog_context.update_program_with_branch(branch_id, branch_node)
            if success:
                curr_context_score = prog_context.get_score()
                if curr_context_score > opt_score:
                    best_top_level_prog_contexts = [prog_context]
                    opt_score = curr_context_score
                elif curr_context_score == opt_score:
                    best_top_level_prog_contexts.append(prog_context)

        return best_top_level_prog_contexts

    def synthesize_branch(self, branch_node: BranchNode, task: Task, examples: PredExample, extractor_enum_lim,
                          pred_enum_lim, extractor_depth_lim, pruning, decomposition) -> BranchNode:

        opt_f1 = 0.0
        branch_program = []
        prog_id = 0
        self.guard_synth.reinit()
        guard_generator = self.guard_synth.get_next_guard(task, examples, pred_enum_lim, pruning)
        while True:
            try:
                guard = next(guard_generator)
                # print("guard:", guard)
            except StopIteration:
                break

            if pruning:
                if self.get_upper_bound(guard, examples.pos_benchmarks) < opt_f1:
                    continue

            self.propagate_examples(guard, examples.pos_benchmarks)

            # print("bp{} propagated_examples: {}".format(prog_id, [ex.section_locator_exec_res for ex in
            #                                                       examples.pos_benchmarks]))

            examples_key = str(examples.pos_benchmarks)
            locator_id = guard.section_locator.get_id()
            if self.branch_synthesized_cache.get(examples_key) is not None and \
                    self.branch_synthesized_cache.get(examples_key).get(locator_id) is not None:
                self.branch_synthesized_cache_hit += 1
                extractors_context, f1 = \
                    self.branch_synthesized_cache.get(examples_key).get(locator_id)
            else:
                self.branch_synthesized_cache_miss += 1
                if decomposition:
                    extractors_context, f1 = self.extract_synth.synthesize(task, examples.pos_benchmarks,
                                                                           extractor_enum_lim,
                                                                           extractor_depth_lim, pruning, opt_f1,
                                                                           locator_prog_id=guard.section_locator.get_id())
                else:
                    extractors_context, f1 = self.extract_synth.synthesize(task, examples.pos_benchmarks,
                                                                           extractor_enum_lim,
                                                                           extractor_depth_lim, False, 0.0,
                                                                           locator_prog_id=guard.section_locator.get_id())
                self.branch_synthesized_cache[examples_key][locator_id] = (extractors_context, f1)

            # print("extractors, f1:", extractors_context, " ", f1)
            # print("opt_f1:", opt_f1)

            if len(extractors_context) > 0:
                if f1 > opt_f1:
                    opt_f1 = f1
                    if decomposition:
                        self.guard_synth.opt_f1 = f1
                    branch_program = [BranchProgram(prog_id, [guard], extractors_context)]
                    # for extractor in extractors_context:
                    #     print('extractor_f1:', extractor.get_stats('f1'))
                elif f1 == opt_f1:
                    branch_program.append(BranchProgram(prog_id, [guard], extractors_context))
                    # for extractor in extractors_context:
                    #     print('extractor_f1:', extractor.get_stats('f1'))

            prog_id += 1

            # if prog_id > 1:
            #     assert False

        if len(branch_program) > 0:
            branch_node.progs = branch_program
            branch_node.f1 = opt_f1
            branch_node.failed = False
        else:
            branch_node.failed = True

        return branch_node

    def propagate_examples(self, guard: GuardProgram, examples: List[ExtractExample]):
        for ex in examples:
            ex.section_locator_exec_res = guard.section_locator.exec_results[ex.name]
            ex.locator_prog_id = guard.section_locator.get_id()

    def get_upper_bound(self, guard: GuardProgram, examples: List[ExtractExample]):
        return guard.section_locator.get_avg_f1()

    def print_cache_stats(self):
        return "cache_hit: {}, cache_miss: {}".format(
            self.branch_synthesized_cache_hit, self.branch_synthesized_cache_miss)
