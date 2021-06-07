import os
import random
import statistics
from typing import List

import run_synth
from .. import config


class TrainingDataAnalysis:
    def __init__(self):
        pass

    def generate_train(self, n: int, domain: str) -> List[int]:
        """
        Select random n benchmarks for training in given domain.
        """
        poss = config.test_benchmarks[domain]
        random.shuffle(poss)
        res = poss[:n]
        return [r[r.index('_')+1:] for r in res]

    def run(self, domain: str, task_id: str,
            num_runs: int = 10, num_train: int = 4):
        f1_scores = []
        for _ in range(num_runs):
            train = self.generate_train(num_train, domain)
            print(train)
            config.task_train_set[domain][task_id][0] = train
            res = run_synth.run_overall_synth(
                domain, task_id, 50, 10000, 100, 7,
                False, False, False, False, True, 'both',
                os.path.join(os.getcwd(), config.PARSED_BENCHMARK_FOLDER),
                verbose=False)
            if res is not None:
                f1_scores.append(res.get_avg_f1())
        print(f1_scores)
        print('min:', min(f1_scores))
        print('avg:', statistics.mean(f1_scores))
        print('max:', max(f1_scores))
        print('quantiles:', statistics.quantiles(f1_scores))
