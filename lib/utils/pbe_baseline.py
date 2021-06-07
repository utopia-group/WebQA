import os
import subprocess
from typing import List

from .. import config
from lib.spec import BaselineExample
from lib.tree import LabelTree
from lib.utils.result_utils import compute_f1


class RunPBEBaseline:
    """
    Class to run the PBE baseline on benchmarks.
    """

    def process_gt(self, benchmark, node_gt):
        if not node_gt:
            return benchmark.gt
        label_tree = LabelTree()
        label_tree.construct_tree(benchmark.pt, benchmark.gt)
        nodes = label_tree.contained_nodes
        new_gt = ''
        for node_id in nodes:
            node = benchmark.pt.get_node_by_tuple_id(node_id)
            if node[1] is not None:
                res = node[1]
            else:
                res = node[0]
            res = benchmark.pt.context_gen(res, qa_context=False)
            # res = benchmark.pt.context_gen(node[0], qa_context=False)
            res = res.replace('"', '')
            res = res.replace('|', '')
            res = res.replace('^', '')
            if len(res) > 0:
                if len(new_gt) > 0:
                    new_gt += '|'
                new_gt += '"' + res + '"'
        return new_gt

    def run_synthesis(
            self, train_benchmarks: List[BaselineExample],
            test_benchmarks: List[BaselineExample],
            benchmark_dir='../../' + config.RAW_BENCHMARK_FOLDER,
            node_gt=False):
        """
        Run C# PBE baseline on the given benchmarks.

        :return: list of outputs for each benchmark
        """
        # Construct arguments to pass to csharp
        args = ['dotnet', 'run', str(len(train_benchmarks)), benchmark_dir]
        used_benchmarks = set()
        for b in train_benchmarks:
            used_benchmarks.add(b.name)
        benchmarks = [
            '{}.html;{}'.format(b.name, self.process_gt(b, node_gt))
            for b in train_benchmarks]
        ordered_benchmarks = [b for b in train_benchmarks]
        for b in test_benchmarks:
            if b.name not in used_benchmarks:
                benchmarks.append('{}.html;{}'.format(
                    b.name, self.process_gt(b, node_gt)))
                ordered_benchmarks.append(b)
        args.extend(benchmarks)

        # Execute synthesizer
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd='./PBE_baseline/Extraction.Web')
        stdout = proc.communicate()[0]
        rc = proc.returncode

        # Parse results
        lines = stdout.decode('utf8').split('\n')[:-1]
        pos = 0
        results = []
        print("lines:", lines)
        for benchmark in ordered_benchmarks:
            res = []
            if rc == 0:
                assert(len(lines[pos]) >= 9 and lines[pos][:9] == 'benchmark')
                pos += 1
                assert(len(lines[pos]) >= 8 and lines[pos][:8] == 'column 1')
                pos += 1
                while pos < len(lines) and \
                        (len(lines[pos]) < 9 or lines[pos][:9] != 'benchmark'):
                    res.append(eval('"' + lines[pos][lines[pos].index(':') + 2:] + '"'))
                    pos += 1

            obj_res = {}
            obj_res['file_name'] = benchmark.name
            obj_res['output'] = ', '.join(res)
            obj_res['precision'], obj_res['recall'], \
                obj_res['f1'], obj_res['na'] = \
                compute_f1(
                    obj_res['output'],
                    self.process_gt(benchmark, node_gt), baseline=True)
            results.append(obj_res)
        return results

    def run(self, domain: str, task_id: str,
            train_benchmarks: List[BaselineExample],
            test_benchmarks: List[BaselineExample],
            node_gt=False):
        """
        Execute PBE synthesis baseline on the train examples and run on test
        set.

        :return: list of results on the test set.
        """
        results = self.run_synthesis(
            train_benchmarks, test_benchmarks, node_gt=node_gt)
        for result in results:
            result['task_id'] = task_id
        return results

    def __init__(self):
        pass
