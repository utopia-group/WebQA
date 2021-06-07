import json
import os
import pathlib
import subprocess
from typing import List

from .. import config
from lib.utils.result_utils import compute_f1
from lib.spec import BaselineExample

class RunACL14Baseline:
    """
    ACL14 Baseline Runner
    """
    def get_path(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../acl14_baseline')
        return path

    def make_dataset(self, domain: str, task_id: str,
                     benchmarks: List[BaselineExample]):
        query, searchquery = config.baseline_tasks[domain][task_id]
        data = []
        for benchmark in benchmarks:
            obj = {
                'query': query,
                'searchquery': searchquery,
                'url': 'fn:{}'.format(benchmark.name),
                'hashcode': '{}.html'.format(benchmark.name),
                'criteria': {
                    'first': 'na',
                    'second': 'na',
                    'last': 'na'
                }
            }
            data.append(obj)
        dataset = {
            'data': data,
            'options': {
                'detailed': True,
                'strict': False,
                'useHashcode': True,
                'cacheDirectory': '../../raw_htmls'
            }
        }
        json_dataset = json.dumps(dataset)
        path = os.path.join(self.get_path(),
            'datasets/webextract/{}_{}.json'.format(domain, task_id))
        with open(path, 'w') as f:
            f.write(json_dataset)
    
    def install_dependencies(self):
        lib = os.path.join(self.get_path(), 'lib')
        print(lib)
        if not os.path.exists(lib):
            dependencies = ['core', 'ling', 'dataset_debug', 'dataset_openweb', 'model']
            for dep in dependencies:
                args = ['./download-dependencies', dep]
                print(args)
                proc = subprocess.Popen(
                    ' '.join(args), cwd='./acl14_baseline',
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    shell=True)
                print(proc.communicate()[1])
            proc = subprocess.Popen('make', cwd='./acl14_baseline',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(proc.communicate()[1])

    def run(self, domain: str, task_id: str,
            benchmarks: List[BaselineExample]):
        self.install_dependencies()
        self.make_dataset(domain, task_id, benchmarks)

        args = [
            './web-entity-extractor', '@memsize=high', '@mode=load',
            '-loadModel', 'models/openweb-devset',
            '@data={}_{}'.format(domain, task_id), '-numThreads', '0']

        output_path = os.path.join(
            self.get_path(), 'webextract.{}_{}.out'.format(domain, task_id))
        pathlib.Path(output_path).touch()

        # Remove python3 from path (little hacky)
        path = os.environ['PATH']
        paths = path.split(':')
        paths = list(filter(lambda p: '/anaconda3/' not in p, paths))
        path = ':'.join(paths)
        new_env = os.environ
        new_env['PATH'] = path
        proc = subprocess.Popen(
            ' '.join(args), cwd='./acl14_baseline',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True, env=new_env)
        stdout, stderr = proc.communicate()
        print('stdout:', stdout)
        print('stderr:', stderr)

        # Parse output
        benchmark_map = {}
        for benchmark in benchmarks:
            benchmark_map[benchmark.name] = benchmark
        res = []
        with open(output_path, 'r') as f:
            lines = f.read().split('\n')
            pos = 0
            # print(lines)
            while pos < len(lines):
                if len(lines[pos]) >= 9 and lines[pos][:9] == 'filename\t':
                    filename = lines[pos][9:]
                    if len(lines[pos + 1]) < 8 or lines[pos + 1][:8] != 'results\t':
                        results = '{}'
                    else:
                        results = lines[pos + 1][8:]
                    benchmark = filename[filename.rindex('[fn:')+4:filename.rindex(']')]
                    results = '[' + results[results.index('{') + 1:results.rindex('}')] + ']'
                    results = json.loads(results)
                    output = ', '.join(results)
                    obj = {}
                    obj['file_name'] = benchmark
                    obj['output'] = output
                    obj['task_id'] = task_id
                    obj['precision'], obj['recall'], \
                        obj['f1'], obj['na'] = \
                        compute_f1(obj['output'], benchmark_map[benchmark].gt, baseline=True)
                    pos += 2
                    res.append(obj)
                else:
                    pos += 1
        return res

    def __init__(self):
        pass
