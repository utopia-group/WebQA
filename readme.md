# WebQA: Web Question Answering with Neurosymbolic Program Synthesis

This is the code repository for the paper ["Web Question Answering with Neurosymbolic Program Synthesis"](https://arxiv.org/abs/2104.07162).

## Pre-requisites

This repository requires to run on Python 3.8.3. Run the following command to install the relevant packages and models:

```
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

We recommend to run the code on machines with GPU, but it should be fine to run on any machines. 


## Benchmark structure

To run any benchmarks, first there needs to exist a benchmark folder `$benchmark_name` in the main directory (the current codebase already includes the benchmarks used in evaluation named `benchmarks`). 

The folder `$benchmark_name` should includes the following folders:
- `raw`: includes the raw html files. The naming of the html follows the pattern `$domain_$bid.txt`. In the evaluation, we use benchmarks from 4 domains: `faculty`, `class`, `conference`, `clinic`. If you only have one domain, still name the `$domain`  to something so that the code can recognize the files. `$bid` is an integer value that gives a unique id to the specific webpage in the domain.
- `gt`: includes the csv file containing the ground truth for each benchmarks. The naming of the csv follows the pattern `$domain_gt.csv`. We explains the structure of the csv file in a later section. 
- `parsed`: includes the parsed benchmarks after running `run_parsing`.

### Structure of the ground truth file:

The ground truth file for each domain is a table with specific benchmark files and its extraction ground truth for a extraction task `t$tid`. You can include ground truth for multiple extraction tasks in the same csv file (and this is recommended). The table looks like the following:

| id   | t$tid  |
| ---- | ------ |
| $bid | GT_STR |
| ...  | ...    |


#### `GT_STR`

`GT_STR` is the ground truth string for benchmark `$bid` for the task `t$tid`. If there is only one ground truth string `GT_STR1`, then 

`GT_STR` =  ""`GT_STR1`""

If there is multiple ground truth string `GT_STR1`, ..., `GT_STRN`, then

`GT_STR` = ""`GT_STR1`""|""`GT_STR2`""|...|""`GT_STRN`""

We will release a script that helps labeling the ground truth and format it in the correct way.


## Define training set and task

Task-related config is defined in `lib/config/run_config.py`. 

Benchmarks files is defined under 'test_benchmarks' variable in the format of `test_benchmarks[$domain] = [list of benchmark file names]`.

Training benchmarks for each task is defined under 'task_train_set' variable in the format of `task_train_set[$domain][t$tid] = [list of benchmark id for training]`

The information related to task is defined under 'tasks' variable in the format of `tasks[$domain][t$tid] = Task(QUESTION, KEYWORDS, CONSTANT_STRINGS)`. `CONSTANT_STRING` and `KEYWORDS` are lists of strings. `KEYWORDS` includes strings that might show up as headers and `CONSTANT_STRINGS` can be set as the same set as `KEYWORDS` unless you think there are some particular strings that you want to contain in the extracted string. 



## Parse benchmarks

The first step given a set of raw html is to parse it into a tree structure by invoking the command

`python3 run_parsing.py parse --benchmark-name $benchmark_name`.

You can specify the specific domain to be parsed by adding the option: `--benchmark $domain` .

You can also specific the specific benchmarks to be parsed by adding the option: `--benchmark ${specific_benchmark_file_name}`.

The parsed benchmarks are stored in `$benchmark_name/parsed`. 


## Run synthesis

Here is the basic command to run the synthesis of benchmarks for a given domain `$domain` and task id `t$tid`

`python3 run.py --run synth --domain $domain --task-id t$tid --ensemble`

The followings are the additional parameter you can play with for synthesis:

- `--ensebmle-sample-size $ensemble_sample_size`: specifies how many programs to sample to construct the ensemble.
- `--repeat-ensemble $repeat_ensemble_times`: specifies how many times the ensemble procedure is run (this computes the variance reduction and average $F_1$ improvement).
- `--disable-pruning`: disables the pruning procedure
- `--disable-decomp`: disables the decomposition between extractor and guard synthesis
- `--extract-enum-lim $extractor_enum_lim`: limits the number of extractors enumerated
- `--extract-dep-lim $extractor_dep_lim`: limits the depth of the extractor enumerated 
- `--pred-enum-lim $guard_enum_lim`: limits the number of guard enumerated
- `--print-time`: print the profiler results at the end of the output (this evaluates the effectiveness of the pruning techniques). 

You should able to see the performance on the benchmarks (i.e., precision, recall and f1) at the end of the output under "ensemble key stats", under the header "best selected p,r,f1".
If you want to see the performance of the ensemble and the its comparison with program selection baselines, you should see "avg f1" and "var f1" under the "our output" and "random avg (var) f1", "shortest avg (var)f1". 
If you enabled `--print_time`, then you should see the profiler output at then end of the file. 

Despite the output, the synthesizer will output two folders:

- Ensemble outputs: `synth_overall_ensemble_results_{your training bids}_test`: 
    - `$domain_t$tid_ensemble_key_stats.txt`: contains the ensemble key stat from the outputs just in case you didn't redirect the output to a file
    - `$domain_t$tid_program_selected_probmass_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: contains the final output program and its extracted outputs on the test set. 
    - `$domain_t$tid_ensemble_context_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: contains the program output by ensemble and its program selection baselines each time the ensemble process is repeated
    - `$domain_t$tid_program_output_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: contains the final output program by the ensemble 
    - `$domain_t$tid_program_results_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: contains the extracted results over test set by the final selected program
    - `$domain_t$tid_program_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: contains the flatted version of the final selected program (i.e. you can interpret using a csv file)
- Training outputs: `synth_overall_results_{your training bids}`:
    - `$domain_t$tid_results_{$top_level_prog_id}_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: The extracted results of the synthesized top level program `$top_level_prog_id` on the training set
    - `$domain_t$tid_program_{$top_level_prog_id}_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: The synthesized program `$top_level_prog_id`. Not that this top level program can represents multiple top level program (in what is described in the paper). We describe the format of the top-level program we output in the following section. 
    - `$domain_t$tid_program_output_{$top_level_prog_id}_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv`: The synthesized program `$top_level_prog_id` and its extracted results on the training set


Note not all the files outputted are meaningful or useful. I only look at the first two files in the ensemble results folder.

### Format of the synthesized top-level program in the output 

Here we are describing the format of the file `$domain_t$tid_program_{$top_level_prog_id}_e{$extractor_enum_lim}_e{$guard_enum_lim}.csv` output by the training process.

For compactness, the top-level program structure that we use in the implementation is a list of branch program where each branch contains a set of guard and a set of extractor. All the guards and extractors inside a branch has the same performance on the training dat. This is the format of the csv file that reflects such structure. 


| P_id      | branch | bp_id | id | program_type | program |
| ------  | ------| ---- | ---- | ---- | ---- |
| top-level program id    | the ith branch inside the top-level program| the branch program id | id for the construct on this row | 'g' stands for this construct is a guard, 'e' stands for extractor | the actual program string |



## User mode

We will release a interface for user to run WebQA (other than running a set of benchmark for evaluation purpose) soon. 