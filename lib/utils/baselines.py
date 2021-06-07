from typing import List

import lib.config as config
from lib.cache import cache
from lib.nlp.qa_bert.run_bert_qa import RunBertQA
from lib.spec import BaselineExample
from lib.utils.benchmark import ProcessBenchmark
from lib.utils.result_utils import compute_f1


class RunQABaseline:
    """
    Class to manage running the QA baseline on benchmarks.
    """

    def __init__(self):
        new_args = {
            'model_name': 'webqa_model',
            # 'model_name': ''
        }
        self.qa = RunBertQA(new_args)
        self.pb = ProcessBenchmark()

    def context_gen(self, file_name, title):
        with open('{}/{}.txt'.format(
                config.PLAIN_TEXTS_FOLDER, file_name), 'r') as input_file:
            file_text = input_file.read().replace('\n', '')
            file_text = title + '\n' + file_text
        return file_text

    def interactive(self):
        """Run baseline with interactive prompts"""
        while True:
            # file_name = input('file_name: ').rstrip()
            file_names = input('file_name: ').rstrip().split(',')
            title = input('title: ').rstrip()
            questions = input('question: ').rstrip()
            # file_text = self.context_gen(file_name, title)
            file_texts = []
            for file_name in file_names:
                file_texts.append(self.context_gen(file_name, title))

            # try:
            section_results, results = self.qa.predict_from_input_squad(file_texts, questions, '0')
            for _, result in results:
                text, qas_all = result[0]
                for (q, a, t, dt, _) in qas_all:
                    print('answer: {}'.format(a))

    @cache(ignore_args=[0, 1])
    def run_benchmark(self, info, domain, task_id, benchmark: BaselineExample):
        """
        Run specific benchmark and execute the QA baseline.
        """
        results = []
        file_text = self.context_gen(benchmark.name, info['title'])
        question = config.tasks[domain][task_id].q
        try:
            _, result = self.qa.predict_from_input_squad(
                [file_text], question, benchmark.name)[1][0]
            text, qas_all = result[0]
            idx = 0
            for (q, a, t, dt, _) in qas_all:
                curr_res = {}
                curr_res['file_name'] = benchmark.name
                curr_res['task'] = task_id
                curr_res['question'] = q
                # only ouptut top-1 here
                curr_res['output'] = a[0]['text']
                if curr_res['output'] == 'empty':
                    curr_res['output'] = ''
                curr_res['dt'] = dt
                curr_res['precision'], curr_res['recall'], \
                    curr_res['f1'], curr_res['na'] = \
                    compute_f1(curr_res['output'], benchmark.gt, baseline=True)
                idx += 1
                results.append(curr_res)
                print("curr_res:", curr_res)
        except Exception as e:
            print('Error while running QA baseline')
            print(e)
        return results

    def run(self, domain, task_id, benchmarks: List[BaselineExample], return_dict=False):
        """
        Run the baseline on a set of benchmarks with a specific task
        and domain.
        """
        benchmark_csv = self.pb.read_benchmarks(domain)
        benchmark_info = {info['id']: info for info in benchmark_csv}
        results = {}
        for benchmark in benchmarks:
            info = benchmark_info[benchmark.name]
            partial = self.run_benchmark(info, domain, task_id, benchmark)
            # print("partial:", partial)
            results[benchmark.name] = partial[0]

        if not return_dict:
            return results.values()
        else:
            return results
