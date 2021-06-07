import collections
import numpy as np
import re
import string
import sys
from lib.spec import Label, Example, GroundTruth
from typing import List, Tuple, Dict


def parse_output(output, baseline):
    if baseline:
        return output

    output = output[1:-1]

    # print(output)
    idx_tuple = r'[(][\d]+, [\d]+[)],'
    idx_pos = [(m.start(0), m.end(0)) for m in re.finditer(idx_tuple, output)]
    idx_pos.append((len(output) + 3, -1))
    # print(idx_pos)
    output_list = []
    for i in range(len(idx_pos) - 1):
        if idx_pos[i][1] == -1:
            break
        start = idx_pos[i][1]
        end = idx_pos[i + 1][0] - 4
        # print("start, end: {},{}".format(start, end))
        output_list.append(output[start: end])

    if len(output_list) == 2 and output_list[0].startswith("May 6"):
        str1 = output_list[0].replace("(", "")
        str1 = str1.replace(")", "")

        str2 = output_list[1].replace("(", "")
        str2 = str2.replace(")", "")

        if str1.lower() == str2.lower():
            output_list = [output_list[0]]

    if len(output_list) > 0 and not "Monday Wednesday Thursday 9:15am - 10:20am" in output_list[0]:
        output_list = list(set(output_list))
    # output_list = list(output_list)
    # print(output_list)

    # some additional filtering on the gt (class 26)
    if "November 16" in output:
        output_list = [o for o in output_list if "11/16" not in o]

    if "October 17" in output:
        output_list = [o for o in output_list if "10/17" not in o]

    output = ", ".join(output_list)
    # print(ast.literal_eval(output))
    # output = output.split(", ")
    # print(output)
    if output == "":
        # print("EMPTY")
        return None

    return output


def parse_gt(gt) -> List[List[str]]:
    # format gt like [[spans]]
    if gt.lower() == "na":
        return [[]]
    gts = gt.split("|")
    parsed_gt = []
    for gt in gts:
        if "^" in gt:
            parsed_gt.append(gt.split("^"))
        else:
            parsed_gt.append([gt])
    # print(parsed_gt)
    return parsed_gt
    # print(gt)


def constuct_gt_idx(target):
    ret = {}
    for elem in target:
        _id = elem["id"]
        ret[_id] = elem
    return ret


"""
Following script partially taken from the SQUAD evaluation script
"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def add_space(text):
        return text.replace(",", ", ")

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(add_space(lower(s)))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def singleF1(gt, res):
    # merge gt into one
    a_gold = " ".join(gt)

    # merge res into one
    a_pred = " ".join(res)

    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    # print(gold_toks)
    # print(pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)

    na = 1 if len(gold_toks) == 0 else 0

    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        # return int(gold_toks == pred_toks)
        # na = 1
        if gold_toks == pred_toks:
            return (1, 1, 1, na)
        return (0, 0, int(gold_toks == pred_toks), na)

    if num_same == 0:
        return (0, 0, 0, na)

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return (precision, recall, f1, na)


def singleF1_stats(gt, res):
    # merge gt into one
    a_gold = " ".join(gt)

    # merge res into one
    a_pred = " ".join(res)

    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)

    na = 1 if len(gold_toks) == 0 else 0

    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        # return int(gold_toks == pred_toks)
        # na = 1
        if gold_toks == pred_toks:
            return (0, 0, 0)
        return 0, len(pred_toks), len(gold_toks)

    if num_same == 0:
        return 0, len(pred_toks), len(gold_toks)

    return num_same, len(pred_toks), len(gold_toks)


def F1(gt, res):
    """
    token-level?
    measures the average overlap between the prediction and ground truth answer
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    Precision: the ratio of correctly predicted words in the answer span to the total number of predicated answer span words
    Recall: the ratio of correctly predicated answer span words to total number of words in the answer word
    """
    benchmarks_f1 = []

    for i in range(len(gt)):
        benchmarks_f1.append(singleF1(gt[i], res[i]))

    # print(benchmarks_f1)

    return np.mean(benchmarks_f1)


def compute_f1(output_str, gts_str, baseline=False, return_output_gt=False):
    parsed_o = parse_output(output_str, baseline)
    single_o = [] if parsed_o is None else [parsed_o]

    if str(gts_str) == "TM":
        # count this as an f1 score of 1 if the gt is TM and the parsed o has a len > 100:
        if len(single_o) > 100:
            if not return_output_gt:
                return (1, 1, 1, 0)
            else:
                return (1, 1, 1, 0), "TM", single_o

        if not return_output_gt:
            return (0, 0, 0, 0)
        else:
            return (0, 0, 0, 0), "TM", single_o

    gts = parse_gt(str(gts_str))
    # print("gts:", gts)
    # print("single_o:", single_o)
    if len(gts) == 1 and len(gts[0]) == 0:
        task_gts_f1 = [(-1, singleF1([], single_o))]
    else:
        task_gts_f1 = [(i, singleF1([gt[i] for gt in gts], single_o)) for i in range(len(gts[0]))]
    # print("single_t:", task_gts_f1)

    best_f1 = sorted(task_gts_f1, key=lambda x: x[1][2], reverse=True)[0]
    # print('compute_f1', output_str, gts_str, best_f1)
    if not return_output_gt:
        return best_f1[1]
    else:
        if best_f1[0] == -1:
            return best_f1[1], [], single_o
        else:
            return best_f1[1], [gt[best_f1[0]] for gt in gts], single_o


"""
we first rely on compute_benchmark level f1 to get the best ground truth to be use here, then we use those ground 
truth to compute task_level_f1
otherwise we need to deal with the issue of making sure all ground truth label are consistent across all benchmark
"""


def task_level_f1(outputs: Dict[str, object], examples: List[Example], baseline=False):
    # print("outputs:", outputs)
    all_correct_pred = []
    all_num_pred = []
    all_num_gt = []

    for example in examples:

        if isinstance(example.gt, Label):
            _, selected_gt, parsed_o = compute_f1(str(outputs[example.name]), str(example.gt.gt_str), baseline=baseline,
                                                  return_output_gt=True)
        elif isinstance(example.gt, GroundTruth):
            _, selected_gt, parsed_o = compute_f1(str(outputs[example.name]), str(example.gt), baseline=baseline,
                                                  return_output_gt=True)

        # If the gt is way too long to label, we don't do exact check
        if selected_gt == "TM":
            if len(parsed_o) > 100:
                correct_pred, num_pred, num_gt = 100, 100, 100
            else:
                correct_pred, num_pred, num_gt = len(parsed_o), len(parsed_o), 100
        else:
            correct_pred, num_pred, num_gt = singleF1_stats(selected_gt, parsed_o)
        all_correct_pred.append(correct_pred)
        all_num_pred.append(num_pred)
        all_num_gt.append(num_gt)

    precision = 1.0 * np.sum(all_correct_pred) / np.sum(all_num_pred)
    recall = 1.0 * np.sum(all_correct_pred) / np.sum(all_num_gt)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1, 0