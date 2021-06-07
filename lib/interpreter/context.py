import collections
import random

import numpy as np
from typing import Tuple, Callable, Dict, List, Set

from lib.tree import HTMLNode
from lib.utils.result_utils import task_level_f1

Subset = collections.namedtuple("Subset", ["bnames", "prog_contexts", "f1"])


class NodeContext:
    def __init__(self, tree_node: HTMLNode, list_node: HTMLNode, match_section_nid):
        self.tree_node: HTMLNode = tree_node
        self.list_node: HTMLNode = list_node

        self.match_section_nid: Tuple[int, int] = match_section_nid

    def __repr__(self):
        return "({},{})".format(self.tree_node, self.list_node)


class StrContext:
    def __init__(self, bindex: Tuple[int, int], str_context, spacy_context, match_section_nid: Tuple[int, int] = (0, 0),
                 partial=False, is_whole_doc=False):
        self.bindex: Tuple[int, int] = bindex
        self.str_context = str_context
        self.spacy_context = spacy_context
        self.partial: bool = partial

        self.is_whole_doc: bool = is_whole_doc
        self.match_section_nid: Tuple[int, int] = match_section_nid

    def get_spacy_text(self, spacy_init: Callable):
        if self.spacy_context is None:
            self.spacy_context = spacy_init(self.str_context)
        return self.spacy_context

    def get_str_text(self):
        if self.str_context is None:
            self.str_context = str(self.spacy_context)
        return self.str_context

    def __repr__(self):
        return "({},{})".format(self.bindex, self.get_str_text())


class PredContext:
    def __init__(self, output: bool, match_section_nids: List[Tuple[int, int]]):
        self.output = output
        self.match_section_nids: Set[Tuple[int, int]] = set(match_section_nids)

    def __repr__(self):
        return "({},{})".format(self.output, self.match_section_nids)


class ResultContext:
    def __init__(self, output, precision=None, recall=None, f1=None, isNA=None, file_name=None, branch_id=None,
                 guard_id=None,
                 extractor_id=None, pred_time=None, prog_time=None):
        self.file_name = file_name
        self.branch_id = branch_id
        self.guard_id = guard_id
        self.extractor_id = extractor_id
        self.output = output
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.isNA = isNA
        self.pred_time = pred_time
        self.prog_time = prog_time

    def duplicate(self):
        return ResultContext(self.output, precision=self.precision, recall=self.recall, f1=self.f1,
                             isNA=self.isNA, file_name=self.file_name, branch_id=self.branch_id, guard_id=self.guard_id,
                             extractor_id=self.extractor_id, pred_time=self.pred_time, prog_time=self.prog_time)

    def update_statistics(self, precision, recall, f1, isNA):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.isNA = isNA

    def field_to_dict(self, pid):
        return_dict = vars(self)
        # return_dict['output_str'] = format_to_print(str(self.output))
        return_dict['P_id'] = str(pid)

        return return_dict

    def __repr__(self):
        return str(self.output)


class SynthProgContext:
    def __init__(self, program, benchmark_results: Dict[str, ResultContext], f1, precision, recall):
        self.program = program
        self.benchmark_results: Dict[str, ResultContext] = benchmark_results
        self.all_f1: List = f1
        self.all_precision: List = precision
        self.all_recall: List = recall
        self.score = None
        self.task_f1 = None

    def output_results(self):
        results = []
        for bres in self.benchmark_results.values():
            results.append(bres.field_to_dict(self.program.get_id()))
        return results

    def output_program_results(self) -> Dict:
        raise NotImplementedError

    def get_score(self):
        raise NotImplementedError

    def get_stats(self, _type="f1", mode="avg", examples=None):

        if _type == "taskf1":
            if self.task_f1 is None:
                self.task_f1 = task_level_f1(dict([(name, res.output) for name, res in self.benchmark_results.items()]),
                                             examples)
            return self.task_f1

        data_match = {'f1': self.all_f1, "precision": self.all_precision, 'recall': self.all_recall}
        mode_match = {'avg': np.mean, 'median': np.median, 'max': np.max, 'min': np.min}

        data = data_match[_type]

        if len(data) == 0:
            return "NA"
        else:
            return mode_match[mode](data)

    def __repr__(self):
        return str(self.output_program_results()) + '{\n' + '\n'.join(
            ['\t' + str(bres) for bres in self.output_results()]) + '}'


class SynthProgContextExt(SynthProgContext):
    def __init__(self, program, benchmark_results: Dict[str, ResultContext], f1=[], precision=[], recall=[]):
        super(SynthProgContextExt, self).__init__(program, benchmark_results, f1=f1, precision=precision, recall=recall)

    def output_program_results(self) -> Dict:
        # print(self.program.exec())
        return {'program_id': self.program.get_id(),
                'program_str': self.program.__repr__(),
                'program': self.program.exec(),
                'd': self.program.depth,
                'f1': self.get_stats('f1'),
                'precision': self.get_stats('precision'),
                'recall': self.get_stats('recall')
                }

    def get_score(self):
        if self.score is None:
            self.score = self.get_stats('f1')
        return self.score


class SynthProgContextTop(SynthProgContext):
    def __init__(self, program, benchmark_results: Dict[str, ResultContext], f1=[], precision=[], recall=[]):
        super(SynthProgContextTop, self).__init__(program, benchmark_results, f1=f1, precision=precision, recall=recall)

    def output_program_results(self) -> Dict:
        return {'program_id': self.program.get_id(),
                'program_str': self.program.__repr__(),
                'program': self.program.exec(),
                'd': self.program.depth,
                'f1_avg': self.get_stats('f1'),
                'precision_avg': self.get_stats('precision'),
                'recall_avg': self.get_stats('recall'),
                'f1_median': self.get_stats('f1', mode='median'),
                'precision_median': self.get_stats('precision', mode='median'),
                'recall_median': self.get_stats('recall', mode='median'),
                'f1_max': self.get_stats('f1', mode='max'),
                'precision_max': self.get_stats('precision', mode='max'),
                'recall_max': self.get_stats('recall', mode='max'),
                'f1_min': self.get_stats('f1', mode='min'),
                'precision_min': self.get_stats('precision', mode='min'),
                'recall_min': self.get_stats('recall', mode='min'),
                'score': self.get_score(),
                'taskf1': self.task_f1}

    def update_program_with_branch(self, branch_id, branch_node):
        # let's just take any program in the branch_node to update the context
        assert len(branch_node.progs) > 0
        extractor_context = branch_node.progs[0].extractor_contexts[0]
        for b_name, b_result in extractor_context.benchmark_results.items():
            b_result_dup = b_result.duplicate()
            b_result_dup.guard_id = branch_node.progs[0].guards[0].get_id()
            b_result_dup.branch_id = branch_id
            self.benchmark_results[b_name] = b_result_dup
            self.all_f1.append(b_result_dup.f1)
            self.all_precision.append(b_result_dup.precision)
            self.all_recall.append(b_result_dup.recall)

    def get_score(self):
        if self.score is None:
            # self.score = self.get_stats('f1')
            if self.program.depth > 2:
               self.score = 0.7 * self.get_stats('f1') + 0.3 * (1 / self.program.depth)
            else:
                # need to penalize cases with low f1, not sure how well this works
            #    if self.get_stats(_type='f1', mode='max') > 0.5 and self.get_stats(_type='f1', mode='min') < 0.05:
                #    self.score = 0.5 * (0.8 * self.get_stats('f1') + 0.2 * (1 / self.program.depth))
            #    else:
                self.score = 0.8 * self.get_stats('f1') + 0.2 * (1 / self.program.depth)
        return self.score


class SynthProgContextGuard(SynthProgContext):
    def __init__(self, program, benchmark_results: Dict[str, ResultContext], pos_total=0, pos_accept=0, neg_total=0,
                 neg_reject=0, pruned=False):
        super(SynthProgContextGuard, self).__init__(program, benchmark_results, f1=[], precision=[], recall=[])
        self.pos_total = pos_total
        self.pos_accept = pos_accept
        self.neg_total = neg_total
        self.neg_reject = neg_reject
        self.pruned = pruned

    def output_program_results(self) -> Dict:
        # print(self.program.exec())
        return {'program_id': self.program.get_id(),
                'program_str': self.program.__repr__(),
                'program': self.program.exec(),
                'd': self.program.depth,
                'pos': (self.pos_accept / self.pos_total) if self.pos_total > 0 else 0,
                'neg': (self.neg_reject / self.neg_total) if self.neg_total > 0 else 0,
                'pruned': self.pruned
                }

    def get_score(self):
        if self.score is None:
            if self.neg_total > 0:
                pos_ratio = (self.pos_accept / self.pos_total) if self.pos_total > 0 else 0
                neg_ratio = (self.neg_reject / self.neg_total) if self.neg_total > 0 else 0
                self.score = (pos_ratio + neg_ratio) / 2
            else:
                pos_ratio = (self.pos_accept / self.pos_total) if self.pos_total > 0 else 0
                self.score = pos_ratio
        return self.score

    def __repr__(self):
        return '\n' + str(self.output_program_results()) + '{\n' + '\n'.join(
            ['\t' + str(bres) for bres in self.output_results()]) + '}'


class EnsembleContext:
    def __init__(self, mode):
        self.ensemble_mode = mode
        self.selected_probmass_prog: List[SynthProgContextTop] = []
        self.selected_softf1_prog: List[SynthProgContextTop] = []
        self.selected_probmass_prog_best_avg = (0, 0, 0)
        self.selected_softf1_prog_best_avg = (0, 0, 0)
        self.selected_probmass_prog_best_task = (0, 0, 0)
        self.selected_softf1_prog_best_task = (0, 0, 0)
        self.best_prog_iter_avg = -1
        self.best_prog_iter_task = -1
        self.all_returned_prog = None
        self.iter_to_stats = {}

    def compute_ith_quantile_prog(self, quantile, sorted_progs: List[SynthProgContextTop], mode, examples):
        if quantile == 0.5:
            if len(sorted_progs) % 2 == 0:
                p1 = sorted_progs[round(len(sorted_progs) / 2) - 1]
                p2 = sorted_progs[round(len(sorted_progs) / 2)]
                if mode == "avg":
                    return np.mean([p1.get_stats('precision'), p2.get_stats('precision')]), np.mean([p1.get_stats(
                        'recall'), p2.get_stats('recall')]), np.mean([p1.get_stats('f1'), p2.get_stats('f1')])
                else:
                    return np.mean([p1.get_stats('taskf1', examples=examples)[0], p2.get_stats('taskf1',examples=examples)[0]]), np.mean([p1.get_stats(
                        'taskf1',examples=examples)[1], p2.get_stats('taskf1',examples=examples)[1]]), np.mean([p1.get_stats('taskf1',examples=examples)[2], p2.get_stats(
                        'taskf1',examples=examples)[2]])

            else:
                p = sorted_progs[round(len(sorted_progs) / 2)]
                if mode == "avg":
                    return p.get_stats('precision'), p.get_stats('recall'), p.get_stats('f1')
                else:
                    return p.get_stats('taskf1', examples=examples)
        else:
            if quantile == 1.0:
                p = sorted_progs[-1]
            else:
                if round(quantile * len(sorted_progs)) >= len(sorted_progs):
                    p = sorted_progs[round(quantile * len(sorted_progs)) - 1]
                else:
                    p = sorted_progs[round(quantile * len(sorted_progs))]
            if mode == "avg":
                return p.get_stats('precision'), p.get_stats('recall'), p.get_stats('f1')
            else:
                return p.get_stats('taskf1', examples=examples)

    def update_context_selected_prog(self, mode, iter, selected_prog, examples, return_prog_contexts):
        avg_stats = (selected_prog.get_stats("precision"),
                     selected_prog.get_stats("recall"),
                     selected_prog.get_stats("f1"))
        task_stats = selected_prog.get_stats("taskf1", examples=examples)

        self.append_iter_to_stats(iter, selected_probmass=selected_prog,
                                    probmass_avg_stats=avg_stats,
                                    probmass_task_stats=selected_prog.get_stats(_type="taskf1",
                                                                                examples=examples))

        self.selected_probmass_prog.append(selected_prog)
        if self.selected_probmass_prog_best_avg[2] < avg_stats[2]:
            self.best_prog_iter_avg = iter
            self.selected_probmass_prog_best_avg = avg_stats
            self.all_returned_prog = return_prog_contexts
        if self.selected_probmass_prog_best_task[2] < task_stats[2]:
            self.best_prog_iter_task = iter
            self.selected_probmass_prog_best_task = task_stats
        
    def update_context_other_progs(self, iter, all_sampled_prog_contexts: List[SynthProgContextTop], examples):

        all_sampled_avg_stats = []
        all_sampled_task_stats = []
        for context in all_sampled_prog_contexts:
            all_sampled_avg_stats.append((context.get_stats('precision'), context.get_stats('recall'),
                                          context.get_stats('f1')))
            all_sampled_task_stats.append(context.get_stats('taskf1', examples=examples))
        # best_prog
        sorted_avg_prog = sorted(all_sampled_prog_contexts, key=lambda x: x.get_stats('f1'), reverse=True)
        sorted_avg_prog_reverse = sorted(all_sampled_prog_contexts, key=lambda x: x.get_stats('f1'))
        self.append_iter_to_stats(iter, best_prog_avg=sorted_avg_prog[0], best_prog_avg_stats=(
            sorted_avg_prog[0].get_stats('precision'),
            sorted_avg_prog[0].get_stats('recall'),
            sorted_avg_prog[0].get_stats('f1')))

        sorted_task_prog = sorted(all_sampled_prog_contexts, key=lambda x: x.get_stats('taskf1', examples=examples)[2],
                                  reverse=True)
        sorted_task_prog_reverse = sorted(all_sampled_prog_contexts, key=lambda x: x.get_stats('taskf1',
                                                                                               examples=examples)[2])
        self.append_iter_to_stats(iter, best_prog_task=sorted_task_prog[0], best_prog_task_stats=
        sorted_task_prog[0].task_f1)

        # avg_prog_avg_stats, median_prog_avg_stats, random_avg_stats,
        self.append_iter_to_stats(iter, avg_prog_avg_stats=(np.mean([x[0] for x in all_sampled_avg_stats]),
                                                            np.mean([x[1] for x in all_sampled_avg_stats]),
                                                            np.mean([x[2] for x in all_sampled_avg_stats])),
                                  random_avg_stats=random.sample(all_sampled_avg_stats, 1)[0],
                                  median_prog_avg_stats=self.compute_ith_quantile_prog(0.50, sorted_avg_prog_reverse,
                                                                                       "avg", examples),
                                  percentile75_avg_stats=self.compute_ith_quantile_prog(0.75, sorted_avg_prog_reverse,
                                                                                        "avg", examples),
                                  min_avg_stats=self.compute_ith_quantile_prog(0, sorted_avg_prog_reverse, "avg",
                                                                               examples),
                                  max_avg_stats=self.compute_ith_quantile_prog(1.0, sorted_avg_prog_reverse, "avg",
                                                                               examples),
                                  avg_prog_task_stats=(np.mean([x[0] for x in all_sampled_task_stats]),
                                                       np.mean([x[1] for x in all_sampled_task_stats]),
                                                       np.mean([x[2] for x in all_sampled_task_stats])),
                                  median_prog_task_stats=self.compute_ith_quantile_prog(0.50, sorted_task_prog_reverse,
                                                                                        "taskf1", examples),
                                  percentile75_task_stats=self.compute_ith_quantile_prog(0.75, sorted_task_prog_reverse,
                                                                                         "taskf1", examples),
                                  min_task_stats=self.compute_ith_quantile_prog(0, sorted_task_prog_reverse, "taskf1",
                                                                                examples),
                                  max_task_stats=self.compute_ith_quantile_prog(1.0, sorted_task_prog_reverse,
                                                                                "taskf1", examples),
                                  random_task_stats=random.sample(all_sampled_task_stats, 1)[0])

        # shortest prog stats
        sorted_prog_by_total_size = sorted(all_sampled_prog_contexts, key=lambda x: x.program.get_avg_prog_size())
        shortest_prog_size = sorted_prog_by_total_size[0].program.get_avg_prog_size()
        shortest_progs = [context for context in all_sampled_prog_contexts if context.program.get_avg_prog_size() ==
                          shortest_prog_size]
        randomly_selected_shortest_prog: SynthProgContextTop = random.sample(shortest_progs, 1)[0]
        self.append_iter_to_stats(iter, shortest_prog=randomly_selected_shortest_prog,
                                  shortest_avg_stats=(randomly_selected_shortest_prog.get_stats('precision'),
                                                      randomly_selected_shortest_prog.get_stats('recall'),
                                                      randomly_selected_shortest_prog.get_stats('f1')),
                                  shortest_task_stats=randomly_selected_shortest_prog.get_stats('taskf1', examples=examples))

    def append_iter_to_stats(self, iter, **args):
        # selected_probmass, probmass_avg_stats, probmass_task_stats, probmass_percentile,
        # selected_softf1, softf1_avg_stats, soft_task_stats, softf1_percentile,

        if self.iter_to_stats.get(iter) is None:
            self.iter_to_stats[iter] = {}
        self.iter_to_stats[iter].update(args)

    def to_output_dict(self):
        new_dict = {}
        for iter, val in self.iter_to_stats.items():
            if new_dict.get(iter) is None:
                new_dict[iter] = {}
            for k, v in self.iter_to_stats[iter].items():
                if k == "selected_probmass" or k == "shortest_prog" or k == "best_prog_avg" or k == "best_prog_task":
                    new_dict[iter][k] = "({}, {})".format(v.program.get_id(), ",".join([str(e) for e in v.program.get_pred_prog_list()]))
                else:
                    new_dict[iter][k] = v
        return new_dict
