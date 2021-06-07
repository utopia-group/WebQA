import re

import numpy as np
import random
from scipy import stats
from collections import Counter, defaultdict
from typing import List, Tuple

from lib.config import PRINT_ENSEMBLE_DETAILS
from lib.evaluator import Evaluator
from lib.interpreter.context import SynthProgContextTop
from lib.program import TopLevelProgram
from lib.spec import Label, ExtractExample, Task
from lib.utils.misc_utils import printc
from lib.utils.result_utils import get_tokens, parse_gt, task_level_f1
from lib.interpreter.context import EnsembleContext


class Ensemble:
    def __init__(self, mode):
        self.evalutor = Evaluator()
        self.mode = mode

    def compute_soft_gt(self, sampled_prog_outputs: List[SynthProgContextTop], bname: str):
        sum_output_tokenize = Counter()
        for single_prog_context in sampled_prog_outputs:
            sum_output_tokenize += Counter(self.tokenize_output(single_prog_context, bname))
        for key in sum_output_tokenize:
            sum_output_tokenize[key] /= len(sampled_prog_outputs)

        return sum_output_tokenize

    def tokenize_output(self, output_context: SynthProgContextTop, bname: str):
        bname_outputs = []
        for output in output_context.benchmark_results[bname].output:
            # print("output:", output)
            bname_outputs.append(output.get_str_text())
        # return get_tokens(" ".join(list(set(bname_outputs))))
        return list(set(get_tokens(" ".join(list(set(bname_outputs))))))

    # TODO: here we only consider a single gt for each benchmark
    def tokenize_gt(self, gt: Label):
        gt_list = []
        for gt in parse_gt(str(gt.gt_str)):
            if len(gt) > 0:
                gt_list.append(gt[0])
        if len(gt_list) == 0:
            return get_tokens("")
        else:
            # return get_tokens(" ".join(gt_list))
            return list(set(get_tokens(" ".join(gt_list))))

    def compute_probmass_helper(self, soft_gt: Counter, output_counter: Counter):
        printc(PRINT_ENSEMBLE_DETAILS, "soft gt: ", soft_gt)
        printc(PRINT_ENSEMBLE_DETAILS, "output_counter: ", output_counter)
        score = 0
        for token, prob in soft_gt.items():
            if output_counter.get(token) is not None:
                score += prob
            else:
                score += 1 - prob
        return score

    def select_prog_with_heuristic(self, best_prog_contexts: List[SynthProgContextTop]):

        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3

        # NOTE: we only consider cases with one branch
        if len(best_prog_contexts[0].program.nodes) > 1:
            return best_prog_contexts[0]

        islist_prog = []
        isany_prog = []

        matchsection_1 = []
        matchsection_2 = []

        prog_id_to_context = {}

        for prog_context in best_prog_contexts:
            assert isinstance(prog_context.program, TopLevelProgram)

            prog_str = prog_context.program.get_branch(list(prog_context.program.nodes.values())[0].id)[1].progs[
                0].guards[0].section_locator.exec()
            prog_id_to_context[prog_context.program.get_id()] = prog_context

            if re.match(re.compile(r".*matchSection\d, w, 1.*"), prog_str) is not None:
                matchsection_1.append(prog_context)
            if re.match(re.compile(r".*matchSection\d, w, 2.*"), prog_str) is not None:
                matchsection_2.append(prog_context)

            if re.match(re.compile(r".*dsl.isAny.*"), prog_str) is not None:
                isany_prog.append(prog_context)
            if re.match(re.compile(r".*dsl.isStructured.*"), prog_str) is not None:
                islist_prog.append(prog_context)

        if len(islist_prog) > 0 and len(isany_prog) > 0:
            if len(matchsection_2) > 0 and len(matchsection_1) > 0:
                intersect_isany_matchsection1 = intersection(isany_prog, matchsection_1)
                if len(intersect_isany_matchsection1) > 0:
                    return prog_id_to_context[intersect_isany_matchsection1[0].program.get_id()]
                else:
                    return prog_id_to_context[isany_prog[0].program.get_id()]
            else:
                return prog_id_to_context[isany_prog[0].program.get_id()]

        if len(matchsection_1) > 0 and len(matchsection_2) > 0:
            return prog_id_to_context[matchsection_1[0].program.get_id()]

        return random.sample(best_prog_contexts, 1)[0]
        # return best_prog_contexts[0]

    def find_best_prog(self, synth_progs: List[SynthProgContextTop], examples: List[ExtractExample], task: Task,
                       sampled_size=1000, repeat=1):

        return_context = EnsembleContext(self.mode)

        for iter in range(repeat):
            sampled_outputs = []

            for i in range(sampled_size):
                # print("{}th sampled program".format(i))
                sampled_top_level = synth_progs[random.randint(0, len(synth_progs) - 1)]
                sampled_outputs.append(
                    self.evalutor.eval_toplevel(task, examples, sampled_top_level.program, idx=str(i)))

            # print("******* sampled outputs *******")
            # for output in sampled_outputs:
            #     print(output.output_program_results())
            # print("*******************************")

            bid_to_soft_gt = {}
            for ex in examples:
                bid_to_soft_gt[ex.name] = self.compute_soft_gt(sampled_outputs, ex.name)
                #print("soft_gt {}: {}".format(ex.name, bid_to_soft_gt[ex.name]))

            prog_to_soft_stats_probmass = {}
            prog_id_to_prog_context = {}

            for output in sampled_outputs:
                prog_id = output.program.get_id()
                if iter == 0:
                    printc(PRINT_ENSEMBLE_DETAILS, "prog id: {}_{}".format(iter, prog_id))
                prog_id_to_prog_context[prog_id] = output

                prog_to_soft_stats_probmass[prog_id] = []

                for ex in examples:
                    if iter == 0:
                        printc(PRINT_ENSEMBLE_DETAILS, "ex {} compute score".format(ex.name))

                    if not self.mode == "f1":
                        prog_ex_stats = self.compute_probmass_helper(bid_to_soft_gt[ex.name],
                                                                     Counter(self.tokenize_output(output, ex.name)))
                        prog_to_soft_stats_probmass[prog_id].append(prog_ex_stats)
                        if iter == 0:
                            printc(PRINT_ENSEMBLE_DETAILS, "stat: {}".format(prog_ex_stats))

                # print("prog_to_soft_stats_probmass:", prog_to_soft_stats_probmass)
                
                if iter == 0:
                    printc(PRINT_ENSEMBLE_DETAILS, "** avg score: ", np.mean(prog_to_soft_stats_probmass[prog_id]))

            prog_output_score_sorted = sorted(prog_to_soft_stats_probmass.items(),
                                                key=lambda x: np.mean(x[1]), reverse=True)
            best_prog_score = prog_output_score_sorted[0][1]
            best_progs_probmass = [prog_id_to_prog_context[prog_id_score[0]] for prog_id_score in
                                    prog_output_score_sorted if prog_id_score[1] == best_prog_score]
            print("best_progs_probmass {}: {}".format(iter, [c.program.exec() for c in best_progs_probmass]))
            selected_prog_probmass = self.select_prog_with_heuristic(best_progs_probmass)
            print("selected_prog_probmass {}: {}".format(iter, selected_prog_probmass.program.exec()))
            selected_prog_probmass = prog_id_to_prog_context[prog_output_score_sorted[0][0]]
            return_context.update_context_selected_prog("probmass", iter, selected_prog_probmass, examples,
                                                        [prog_id_to_prog_context[context[0]] for
                                                            context in prog_output_score_sorted])

            return_context.update_context_other_progs(iter, list(prog_id_to_prog_context.values()), examples)

            # self.find_uncertain_benchmark(sampled_outputs)

        return return_context
