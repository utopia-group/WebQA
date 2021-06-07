
"""
This file stores some general config values for scripts, but not specific
to running tasks as those are currently stored in run_config.py.
"""

BENCHMARK_DOMAINS = ['fac', 'conf', 'class']

RAW_BENCHMARK_FOLDER = 'raw'
PARSED_BENCHMARK_FOLDER = 'parsed'
GT_FOLDER = 'gt'
BENCHMARK_INFO_FOLDER = 'benchmark_list'
SYNTH_OUTPUT_FOLDER = 'synth_results'
SYNTH_EXT_OUTPUT_FOLDER = 'synth_ext_results'
SYNTH_PRED_OUTPUT_FOLDER = 'synth_pred_results'
SYNTH_OVERALL_OUTPUT_FOLDER = 'synth_overall_results'
SYNTH_OVERALL_ENSEMBLE_FOLDER = 'synth_overall_ensemble_results'
SYNTH_OVERALL_ACTIVE_FOLDER = 'synth_overall_active_results'
OVERALL_OUPUT_FOLDER = 'overall_prog_outputs'
BASELINE_RESULTS_FOLDER = 'baselines_res'
PLAIN_TEXTS_FOLDER = 'plain_texts'
PARSER_EXECUTABLE = 'webextract-parse'

CACHE_PATH = '.cache'
CACHE_IN_MEMORY = True
IN_MEMORY_CACHE_SIZE = 500000

PRINT_MATCHSECTION_INFO = False
PRINT_ENSEMBLE_DETAILS = False
PRINT_EVAL_DETAILS = False

PARTIAL_EXACT_MATCH = False

def set_param(args):
    global PARTIAL_EXACT_MATCH
    if args.domain == 'clinic' and (args.task_id == 't2' or args.task_id == 't3'):
        PARTIAL_EXACT_MATCH = True
        print("set_param:", PARTIAL_EXACT_MATCH)

def read_flag(op):
    if op == 'partial_exact_match':
        return PARTIAL_EXACT_MATCH


def set_cache_path(new_cache_path):
    global CACHE_PATH
    CACHE_PATH = new_cache_path
