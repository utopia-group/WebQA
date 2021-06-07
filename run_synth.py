import os
import time
import cProfile, pstats, io
from pstats import SortKey
import numpy as np
import pandas as pd
import lib.config as config
import lib.grammar.constant as constant
from lib.config import tasks, task_train_set, test_benchmarks, MANUAL_OVERALL_PROGRAM, OVERALL_OUPUT_FOLDER
from lib.ensemble import Ensemble
from lib.top_level_synthesizer import TopLevelSynthesizer
from lib.interpreter.context import ResultContext
from lib.interpreter.dsl import DSL
from lib.evaluator import Evaluator
from lib.spec import Task
from lib.utils.csv_utils import read_csv_to_dict, save_dict_to_csv
from lib.utils.misc_utils import format_to_print, find_date, mkdir, read_benchmarks
from lib.utils.result_utils import constuct_gt_idx, parse_gt, task_level_f1


def format_result_output(args, result_context: ResultContext, task: Task):
    return {'file_name': result_context.file_name,
            'question': task.q,
            'keywords': task.keyword,
            'constants': task.const_str,
            'task_id': args.task_id,
            'program_id': result_context.extractor_id,
            'output': format_to_print(str(result_context.output)),
            'time': result_context.prog_time,
            'f1': result_context.f1,
            'precision': result_context.precision,
            'recall': result_context.recall}


def run_overall_synth(args, verbose=True):
    OUTPUT_FOLDER_TRAIN = '{}_{}'.format(
        config.SYNTH_OVERALL_OUTPUT_FOLDER, '_'.join(
            facn for facn in task_train_set[args.domain][args.task_id][0]))
    if not os.path.exists(OUTPUT_FOLDER_TRAIN):
        os.system('mkdir -p {}'.format(OUTPUT_FOLDER_TRAIN))

    benchmarks = read_benchmarks(args, ['{}_{}'.format(args.domain, facn)
                                        for facn in task_train_set[args.domain][args.task_id][0]],
                                 locate_gt=args.locate_gt, construct_additonal_mapping=False)
    task = tasks[args.domain][args.task_id]
    augment_keyword(task)
    compute_valid_ent_tags(args.benchmark_folder, args.domain, args.task_id, benchmarks)

    config.set_param(args)

    if verbose:
        print("Training...")
    pr = cProfile.Profile()
    pr.enable()

    synth = TopLevelSynthesizer()
    synth_start = time.perf_counter()
    synth_start_clock = time.process_time()
    synth_prog_contexts = synth.synthesize(task, benchmarks, args.extract_enum_lim, args.pred_enum_lim, 
                            args.extract_dep_lim, (not args.disable_pruning), (not args.disable_decomp))
    synth_end = time.perf_counter()
    synth_end_clock = time.process_time()

    for synth_prog_context in synth_prog_contexts:
        program_filepath = '{}/{}_{}_program_{}_e{}_e{}.csv'.format(
            OUTPUT_FOLDER_TRAIN, args.domain, args.task_id, synth_prog_context.program.get_id(),
            args.extract_enum_lim, args.pred_enum_lim)
        save_dict_to_csv(program_filepath,
                         synth_prog_context.program.get_pred_prog_list())

        program_output_filepath = '{}/{}_{}_program_output_{}_e{}_e{}.csv'.format(
            OUTPUT_FOLDER_TRAIN, args.domain, args.task_id, synth_prog_context.program.get_id(),
            args.extract_enum_lim, args.pred_enum_lim)
        with open(program_output_filepath, "w") as o:
            o.write(repr(synth_prog_context))

        results_output_filepath = '{}/{}_{}_results_{}_e{}_e{}.csv'.format(
            OUTPUT_FOLDER_TRAIN, args.domain, args.task_id, synth_prog_context.program.get_id(),
            args.extract_enum_lim, args.pred_enum_lim)
        save_dict_to_csv(results_output_filepath, synth_prog_context.output_results())

    print("training info:")
    print("extractor synthesis pulled states: {}".format(synth.extract_synth.total_explored_states))
    print("guard synthesis pulled states: {}".format(synth.guard_synth.guard_explored))

    if verbose:
        print(f'overall train time: {synth_end - synth_start:0.4f} seconds')
        print(f'overall cpu train time: {synth_end_clock - synth_start_clock:0.4f} seconds')
    pr.disable()

    res = None
    # test directly
    if not args.disable_test and len(synth_prog_contexts) > 0:
        if verbose:
            print("Testing...")
        test_start = time.perf_counter()

        if not args.ensemble:
            OUTPUT_FOLDER_TEST = '{}_{}_test'.format(
                config.SYNTH_OVERALL_OUTPUT_FOLDER, '_'.join(
                    facn for facn in task_train_set[args.domain][args.task_id][0]))
            if not os.path.exists(OUTPUT_FOLDER_TEST):
                os.system('mkdir -p {}'.format(OUTPUT_FOLDER_TEST))

            all_test_benchmarks = read_benchmarks(
                None, test_benchmarks[args.domain], locate_gt=args.locate_gt,
                domain=args.domain, benchmark_folder=args.benchmark_folder, task_id=args.task_id)
            evaluator = Evaluator()
            all_test_prog_results = []
            all_test_program_pred_prog = []
            all_test_outputs = []

            for synth_prog_context in synth_prog_contexts:
                for i in range(25):
                    result = evaluator.eval_toplevel(task, all_test_benchmarks, synth_prog_context.program, idx=str(i))

                    all_test_prog_results.append(result.output_program_results())
                    all_test_program_pred_prog.extend(result.program.get_pred_prog_list())
                    all_test_outputs.extend(result.output_results())

            program_filepath = '{}/{}_{}_program_e{}_e{}.csv'.format(
                OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                args.extract_enum_lim, args.pred_enum_lim)
            save_dict_to_csv(program_filepath, all_test_program_pred_prog)

            results_output_filepath = '{}/{}_{}_program_results_e{}_e{}.csv'.format(
                OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                args.extract_enum_lim, args.pred_enum_lim)
            save_dict_to_csv(results_output_filepath, all_test_outputs)

            program_output_filepath = '{}/{}_{}_program_output_e{}_e{}.csv'.format(
                OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                args.extract_enum_lim, args.pred_enum_lim)
            save_dict_to_csv(program_output_filepath, all_test_prog_results)
            if verbose:
                print(f'overall test time: {synth_end - synth_start:0.4f} seconds')
        else:
            OUTPUT_FOLDER_TEST = '{}_{}_test'.format(
                config.SYNTH_OVERALL_ENSEMBLE_FOLDER, '_'.join(
                    facn for facn in task_train_set[args.domain][args.task_id][0]))
            if not os.path.exists(OUTPUT_FOLDER_TEST):
                os.system('mkdir -p {}'.format(OUTPUT_FOLDER_TEST))

            ensemble = Ensemble('probmass')
            train_benchmarks = ["{}_{}".format(args.domain, b) for b in task_train_set[args.domain][args.task_id][0]]
            only_test_benchmarks = [b for b in test_benchmarks[args.domain] if not b in train_benchmarks]
            if verbose:
                print("only test benchmarks:", only_test_benchmarks)
            all_test_benchmarks = read_benchmarks(
                args, only_test_benchmarks, locate_gt=False, domain=args.domain, benchmark_folder=args.benchmark_folder, task_id=args.task_id, construct_additonal_mapping=False)

            # infer sample size
            total_program_size = 0
            for prog_context in synth_prog_contexts:
                curr_prog = prog_context.program
                branch_prog_accumulate = 1
                for i in curr_prog.exec_order:
                    _, branch_node = curr_prog.get_branch(i)
                    # construct flatten prog version
                    branch_node.sample_random_prog(sample=False)
                    branch_prog_accumulate *= len(branch_node.flatten_progs)
                total_program_size += branch_prog_accumulate
            
            sample_size = 0
            if total_program_size < 50:
                sample_size = total_program_size
            elif total_program_size < 10:
                sample_size = 10
            elif total_program_size < 100:
                sample_size = 50
            elif total_program_size < 200:
                sample_size = 100
            else:
                sample_size = 1000
            print("total_prog_size:", total_program_size)
            print("sample_size:", sample_size)

            returns = ensemble.find_best_prog(synth_prog_contexts, all_test_benchmarks, task,
                                              sampled_size=sample_size,
                                              repeat=args.repeat_ensemble)

            if returns is not None:

                return_progs = returns.all_returned_prog

                all_test_prog_results = []
                all_test_program_pred_prog = []
                all_test_outputs = []

                for context in return_progs:
                    all_test_prog_results.append(context.output_program_results())
                    all_test_program_pred_prog.extend(context.program.get_pred_prog_list())
                    all_test_outputs.extend(context.output_results())

                program_filepath = '{}/{}_{}_program_e{}_e{}.csv'.format(
                    OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                    args.extract_enum_lim, args.pred_enum_lim)
                save_dict_to_csv(program_filepath, all_test_program_pred_prog)

                results_output_filepath = '{}/{}_{}_program_results_e{}_e{}.csv'.format(
                    OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                    args.extract_enum_lim, args.pred_enum_lim)
                save_dict_to_csv(results_output_filepath, all_test_outputs)

                program_output_filepath = '{}/{}_{}_program_output_e{}_e{}.csv'.format(
                    OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                    args.extract_enum_lim, args.pred_enum_lim)
                save_dict_to_csv(program_output_filepath, all_test_prog_results)

                ensemble_context_filepath = '{}/{}_{}_ensemble_context_e{}_e{}.csv'.format(
                    OUTPUT_FOLDER_TEST, args.domain, args.task_id,
                    args.extract_enum_lim, args.pred_enum_lim)
                df = pd.DataFrame(returns.to_output_dict())
                # print(returns.to_output_dict())
                df.to_csv(ensemble_context_filepath)

                probmass_selected = returns.selected_probmass_prog[returns.best_prog_iter_task]
                selected_prog_probmass = "{}/{}_{}_program_selected_e{}_e{}.csv".format(OUTPUT_FOLDER_TEST,
                                                                                                args.domain,
                                                                                                args.task_id,
                                                                                                args.extract_enum_lim,
                                                                                                args.pred_enum_lim)
                with open(selected_prog_probmass, 'w') as f:
                    f.write(str(probmass_selected))

                ensemble_output_text = ""

                ensemble_output_text += "***** ensemble key stats *****\n"
                ensemble_output_text += "===== best selected =====\n"

                ensemble_output_text += "best selected {}: \n{}\n".format(
                    probmass_selected.program.get_id(), probmass_selected.program.get_pred_prog_list())
                ensemble_output_text += "best selected p,r,f1: {}\n\n".format(
                    probmass_selected.task_f1)

                ensemble_output_text += "\n-> in comparsion with\n"

                ensemble_output_text += "median stats for best selected: {}\n".format(returns.iter_to_stats[
                                                                                               returns.best_prog_iter_task][
                                                                                               'median_prog_task_stats'])

                ensemble_output_text += "75 quantile stats for best selected: {}\n".format(returns.iter_to_stats[
                                                                                                    returns.best_prog_iter_task][
                                                                                                    'percentile75_task_stats'])

                ensemble_output_text += "best stats for best selected: {}\n".format(returns.iter_to_stats[
                                                                                             returns.best_prog_iter_task][
                                                                                             'best_prog_task_stats'])

                ensemble_output_text += "===== overall =====\n"
                ensemble_output_text += "overall best f1: {}\n".format(np.max([x['max_task_stats'][2] for x in
                                                                                    returns.iter_to_stats.values()]))
                ensemble_output_text += "overall worst f1: {}\n".format(np.min([x['min_task_stats'][2] for x in
                                                                                     returns.iter_to_stats.values()]))

                ensemble_output_text += "\n-> our output\n\n"

                probmass_selected_all_task_f1 = [x.get_stats('taskf1')[2] for x in returns.selected_probmass_prog]
                ensemble_output_text += "probmass avg task f1: {}\n".format(np.mean(probmass_selected_all_task_f1))
                ensemble_output_text += "probmass var task f1: {}\n".format(np.var(probmass_selected_all_task_f1))

                ensemble_output_text += "\n"

                ensemble_output_text += "\n-> in comparsion with\n\n"
                randomly_sampled_all_task_f1 = [x['random_task_stats'][2] for x in returns.iter_to_stats.values()]
                ensemble_output_text += "random avg f1: {}\n".format(np.mean(randomly_sampled_all_task_f1))
                ensemble_output_text += "random var f1: {}\n".format(np.var(randomly_sampled_all_task_f1))

                ensemble_output_text += "\n"

                shortest_sampled_all_task_f1 = [x['shortest_task_stats'][2] for x in returns.iter_to_stats.values()]
                ensemble_output_text += "shortest avg f1: {}\n".format(np.mean(shortest_sampled_all_task_f1))
                ensemble_output_text += "shortest var f1: {}\n".format(np.var(shortest_sampled_all_task_f1))

                ensemble_output_text += "******************************\n"

                print(ensemble_output_text)

                ensemble_key_stats_filepath = '{}/{}_{}_ensemble_key_stats.txt'.format(
                    OUTPUT_FOLDER_TEST, args.domain, args.task_id)
                with open(ensemble_key_stats_filepath, "w") as f:
                    f.write(ensemble_output_text)

        test_end = time.perf_counter()
        if verbose:
            print(f'overall test time: {test_end - test_start:0.4f} seconds')

    if args.print_time:
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        if verbose:
            print(s.getvalue())

    return res


def run_manual_overall_prog(args):
    config.set_param(args)
    mkdir(OVERALL_OUPUT_FOLDER)
    evaluator = Evaluator()

    overall_prog = MANUAL_OVERALL_PROGRAM[args.domain][args.task_id]
    task = tasks[args.domain][args.task_id]
    benchmarks = read_benchmarks(args, test_benchmarks[args.domain])

    result = evaluator.eval_toplevel(task, benchmarks, overall_prog)
    print("task-level f1: {}".format(task_level_f1(dict([(name, res.output) for name,res in result.benchmark_results.items()]), benchmarks)))

    output_file = "{}/{}_{}_{}_output.csv".format(OVERALL_OUPUT_FOLDER, args.domain, args.task_id,
                                                  overall_prog.id)
    save_dict_to_csv(output_file, result.output_results())
    print("overall f1: ", np.mean(result.all_f1))


def compute_valid_ent_tags(benchmark_folder, domain, task_id, benchmarks, dsl=DSL()):

    target_csv = "{}/gt/{}_gt.csv".format(benchmark_folder, domain)
    target = read_csv_to_dict(target_csv)
    target = constuct_gt_idx(target)

    new_labels = set()
    for b in benchmarks:
        gts = parse_gt(target[b.name][task_id])
        # TODO: here we only analyze minimum gt
        # since some other format gt is not ready
        gts_filtered = []
        for gt in gts:
            if len(gt) > 0:
                gts_filtered.append(gt)
        gt_string = dsl.nlp_api.spacy_init(
            '    '.join([gt[0] for gt in gts_filtered]))

        for ent_l in constant.PRED_ENTITY:
            if ent_l in [e.label_ for e in gt_string.ents]:
                new_labels.add(ent_l)
            if ent_l == 'DATE' or ent_l == 'TIME':
                find_date_res = find_date(str(gt_string), disable_clean_date=True)
                if find_date_res is not None and len(find_date_res) > 0:
                    new_labels.add(ent_l)
        if len(list(gt_string.noun_chunks)) > 0:
            new_labels.add('NOUN')
    constant.update_pred_entity(list(new_labels))
    print('valid entity labels: ', constant.PRED_ENTITY)


# augment the keyword in some scenario
def augment_keyword(task, dsl=DSL()):
    current_keywords = task.keyword
    new_keyword = []
    for keyword in current_keywords:
        phrase_split = keyword.split(" ")
        if len(phrase_split) == 1 and "-" in keyword:
            new_keyword.append(keyword.replace('-', ' '))
        if len(phrase_split) == 1 and keyword.endswith('s') and keyword[:-1].isupper():
            new_keyword.append(keyword[:-1])
        if len(phrase_split) == 1 and keyword.endswith('ments'):
            new_keyword.append(keyword.replace('ments', ''))
            new_keyword.append(keyword.replace('ments', 'ed'))
        if len(phrase_split) == 2:
            context = dsl.nlp_api.spacy_init(keyword)
            # get pos tags
            pos_tags = [token.pos_ for token in context]
            tags = [token.tag_ for token in context]
            # print(pos_tags)
            # print(tags)
            if (pos_tags[0] == 'PROPN' and pos_tags[1] == 'PROPN') and (tags[0] == 'NNP' and tags[1] == 'NNPS'):
                new_keyword.append(phrase_split[1])
            if len(current_keywords) == 1 and pos_tags[0] == 'PRON' and pos_tags[1] == 'NOUN':
                new_keyword.append(phrase_split[1])
            if pos_tags[0] == "VERB" and pos_tags[1] == "NOUN":
                new_keyword.append(phrase_split[1])
            if pos_tags[0] == "NOUN" and pos_tags[1] == "PROPN":
                if not keyword.endswith('s'):
                    new_keyword.append("{}s".format(keyword))
        if len(phrase_split) == 3:
            context = dsl.nlp_api.spacy_init(keyword)
            compound_tokens = [str(token) for token in context if token.dep_ == 'compound']
            new_keyword.append("{}".format(" ".join(compound_tokens)))
            new_keyword.append("{}s".format(" ".join(compound_tokens)))
        new_keyword.append(keyword)
        task.keyword = new_keyword
