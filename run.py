import argparse
import os
import random

import lib.config as config
import run_synth
from lib.cache import set_cache_dir, set_disable_cache, set_update_cache
from lib.utils.experiments import TrainingDataAnalysis

"""
Script in order to execute programs, test json parser, and run synthesizer.
"""

# Setup argparse
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-folder', type=str, default='benchmarks', help='location of the benchmark folder')
parser.add_argument(
    '--domain', type=str,
    help='benchmark domains to use')
parser.add_argument(
    '--task-id', type=str, default='t1')
parser.add_argument(
    '--benchmarks', type=str,
    help='comma separated list of benchmarks to run on instead'
         ' of all in domain')
parser.add_argument(
    '--program-ids', type=str,
    help='comma separated list of program ids (e.g. p1,p2)')
parser.add_argument(
    '--run', type=str,
    help='the task to run')
parser.add_argument(
    '--disable-cache', action='store_true', default=False,
    help='whether to disable the cache while running including updates')
parser.add_argument(
    '--update-cache', action='store_true', default=False,
    help='whether to update the cache and not use previously cached values')
parser.add_argument(
    '--delete-cache', action='store_true', default=False,
    help='whether to delete existing cache (this option is used to run ablation)')
parser.add_argument('--disable-test', action='store_true', default=False,
                    help='whether to disable testing')
parser.add_argument(
    '--ensemble', action='store_true', default=False,
    help="whether to enable ensemble mode on the test data")

parser.add_argument('--ensebmle-sample-size', type=int, default=1000, help='how many program to sample in ensemble')
parser.add_argument('--repeat-ensemble', type=int, default=1, help='repeat ensemble to run ensemble ablation')
parser.add_argument(
    '--disable-pruning', action='store_true', default=False,
    help='whether to disable pruning')
parser.add_argument(
    '--print-time', action='store_true', default=False,
    help='whether to print the overall runtime at the end of the output')
parser.add_argument('--disable-decomp', action='store_true', default=False, help='whether to disable the '
                                                                                        'decomposition synthesis '
                                                                                        'feature')
parser.add_argument(
    '--locate-gt', action='store_true', default=False,
    help="whether to enable the locate gt functionality")
parser.add_argument(
    '--extract-enum-lim', type=int, default=5000)
parser.add_argument(
    '--pred-enum-lim', type=int, default=20000)
parser.add_argument(
    '--extract-dep-lim', type=int, default=5)
parser.add_argument(
    '--enum-size-lim', type=int, default=50)

# Parse arguments
args = parser.parse_args()

print("args:", args)

random.seed(2333)

if args.delete_cache:
    # os.system("rm -r .cache")
    if args.disable_pruning:
        new_cache_path = ".cache_{}_{}_{}".format(args.domain, args.task_id, "no_pruning")
    elif args.disable_decomp:
        new_cache_path = ".cache_{}_{}_{}".format(args.domain, args.task_id, "no_decomp")
    else:
        new_cache_path = ".cache_{}_{}_{}".format(args.domain, args.task_id, "pruning")

    set_cache_dir(new_cache_path)

# Configure cache
set_disable_cache(args.disable_cache)
set_update_cache(args.update_cache)

if args.run == 'exec':
    run_synth.run_manual_overall_prog(args)
    exit()

if args.run == "synth":
    run_synth.run_overall_synth(args)
    exit()

if args.run == 'training-data-analysis':
    TrainingDataAnalysis().run(args.domain, args.task_id)
    exit()

if args.delete_cache:
    os.system("rm -r {}".format(new_cache_path))
    exit()
