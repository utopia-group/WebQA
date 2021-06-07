import datetime

import numpy as np
import os
import re
from typing import List
from dateparser.search import search_dates
import pyap

from lib.json_parser import JsonParser
from lib.spec import BaselineExample, ExtractExample, Label, GroundTruth
from lib.tree import LabelTree, ParseTree
from lib.utils.csv_utils import read_csv_to_dict
from lib.utils.result_utils import constuct_gt_idx

# time_pattern = re.compile(r"(.*\d{1,2}[:]\d{2}.*)|(.*\d - \d)")
time_pattern = re.compile(r"(.*\d{1,2}[:]\d{2}.*)")
remove_md_syntax_re = re.compile(r"(.*)[[]([^]]+)[]][ ]*[(][^)]+[)](.*)")
digit_1_3_5 = re.compile(r"^\d$|^\d{3}$|^\d{5}$")
filter_out_pattern_1 = re.compile(r"^\d \d{1,2}$")
add_hoc_date_pattern = re.compile(r"((January|February|March|April|May|June|July|August|September|October|November"
                                  r"|December) \d{1,2})")
add_hoc_date_pattern_2 = re.compile(r"(\d{1,2}[/]\d{1,2}([/]\d{2,4})?[, ]+\d{1,2}[:]\d{1,2}(am|pm)([ -]?\d{1,2}[:]\d{1,"
                                    r"2}(am|pm))?)")
add_hoc_date_pattern_3 = re.compile(r"((Monday|Tuesday|Wednesday|Thursday|Friday)( ("
                                    r"Monday|Tuesday|Wednesday|Thursday|Friday))*([:])?[ ]\d{1,2}([:]\d{1,"
                                    r"2}(am|AM|pm|PM)?)?([ ]["
                                    r"-][ ]\d{1,2}([:]\d{1,2}(am|AM|pm|PM)?)?)?)")
add_hoc_date_pattern_4 = re.compile(r"((Monday|Tuesday|Wednesday|Thursday|Friday) (\d{1,2} ("
                                    r"January|February|March|April|May|June|July|August|September|October|November"
                                    r"|December))([,]?[ ]?\d{1,2}[:]\d{1,2} - \d{1,2}[:]\d{1,2})?)")
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
clean_string_pattern_1 = re.compile(r"Exam[ ]*\d[ ]*:")

filter_const_getcsv_pattern = re.compile(r".*dsl[.]Filter[(]dsl[.]Split[(].*[)], lambda x[:] dsl[.]hasString.*[)]")
split_str_pattern = re.compile(r"[.]|[,]|[(]|[)]|[ ]|[']")

location_pattern = re.compile(r".*St|Dr|Ave|Street|Drive|Avenue.*")


def printc(condition, *args):
    if condition:
        print(*args)


def mkdir(folder):
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))


def get_file_path(bfolder) -> str:
    return os.path.join(os.getcwd(), bfolder, 'parsed')


def get_pt(file_path, file_name) -> ParseTree:
    jp = JsonParser()
    pt = jp.run_parser(file_path, file_name)
    return pt


def get_gt_file(args, domain=None, benchmark_folder=None):
    if args is not None:
        domain = args.domain
        benchmark_folder = args.benchmark_folder
    target_csv = "{}/{}/{}_gt.csv".format(benchmark_folder, 'gt', domain)
    target = read_csv_to_dict(target_csv)
    target = constuct_gt_idx(target)
    return target


def read_baseline_benchmarks(
        args, benchmarks: List[str]) -> List[BaselineExample]:
    target = get_gt_file(args)
    ret_benchmarks = []
    for b_file_name in benchmarks:
        ret_benchmarks.append(BaselineExample(
            b_file_name, GroundTruth(target[b_file_name][args.task_id]),
            pt=get_pt(get_file_path(args.benchmark_folder), b_file_name)))
    return ret_benchmarks


def read_benchmarks(
        args, benchmarks: List[str], locate_gt=False, domain=None, benchmark_folder=None,
        task_id=None, construct_additonal_mapping=False, find_best_label=False) -> List[ExtractExample]:
    if args is not None:
        domain = args.domain
        task_id = args.task_id
        benchmark_folder = args.benchmark_folder
    file_path = get_file_path(benchmark_folder) if benchmark_folder is not None else get_file_path('benchmarks')
    target = get_gt_file(args, domain=domain, benchmark_folder=benchmark_folder)
    ret_benchmarks = []
    for b_file_name in benchmarks:
        print("b_file_name:", b_file_name)
        pt = get_pt(file_path, b_file_name)

        if construct_additonal_mapping:
            pt.construct_node_parent_mapping((pt.start_node.id, 0), [])

        gt_str = GroundTruth(target[b_file_name][task_id])
        gt_tree = LabelTree()
        if (domain == 'conf' and (task_id == 't1' or task_id == 't3' or task_id == "t5")) or \
                (domain == 'class' and (task_id == 't2')) or \
                (domain == 'clinic' and (task_id == 't5')):
            gt_tree.construct_tree(pt, gt_str, find_best_label=find_best_label, include_duplicate=True)
            if (domain == "conf" and (task_id == 't1' or task_id == "t5")) or \
                    (domain == 'class' and (task_id == 't2')) or \
                    (domain == 'clinic' and (task_id == 't5')):
                gt_tree.gt_nodes_any_correct = True
        elif (domain == 'fac' and (task_id == 't7')) or \
                (domain == 'conf' and (task_id == 't7' or task_id == 't9')) or \
                (domain == 'clinic' and (task_id == 't2')):
            gt_tree.construct_tree(pt, gt_str, find_best_label=find_best_label, not_fine_grained=True)
        else:
            gt_tree.construct_tree(pt, gt_str, find_best_label=find_best_label)
        gt = Label(gt_str, gt_tree, locate_gt)
        ret_benchmarks.append(ExtractExample(b_file_name, gt, pt))

        best_label_tree, gt_nodes = gt_tree.refine_duplicates()
        # print(best_label_tree)
        # print(gt_nodes)

    return ret_benchmarks


def format_to_print(prog_str):
    prog_str = prog_str.replace("\"", "")
    prog_str = prog_str.replace("\n", "")
    prog_str = prog_str.replace("\t", "")

    prog_str = re.sub(' +', ' ', prog_str)

    return prog_str


def remove_md_syntax(content):
    if "[" in content and "]" in content and "(" in content and ")" in content:
        result = re.sub(remove_md_syntax_re, r"\1\2\3", content)
        return result
    else:
        return content


def compute_n_gram_idx(token_idx, n, max_length):
    return [
        (token_idx - (n - 1) + i, token_idx + 1 + i) for i in range(n) if
        (token_idx - (n - 1) + i) >= 0 and (token_idx + 1 + i) <= max_length]


# TODO: a **very** stupid way to get the n-gram but nvm
def get_n_gram(token_idx: int, context_tokenized: List, max_phrase_len: int):
    n_grams = []
    if max_phrase_len < 3:
        n_grams.append(str(context_tokenized[token_idx]))
    if max_phrase_len > 1:
        n_grams.extend([
            str(context_tokenized[s_i:e_i]) for s_i, e_i in
            compute_n_gram_idx(token_idx, 3, len(context_tokenized))])
    n_grams.extend([str(context_tokenized[s_i:e_i]) for s_i, e_i in
                    compute_n_gram_idx(token_idx, 2, len(context_tokenized))])

    return n_grams


def get_priority_depth(p):
    return p.depth


def get_priority_depth_reverse(p):
    return -p.depth


def get_priority_f1_reverse(p):
    return -p.get_avg_f1(approx=False)


def get_priority_f1(p):
    return np.mean(p.all_f1)


def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def clean_context_string(string):
    check_text_str = string
    check_text_str = check_text_str.replace(":", " ")
    check_text_str = check_text_str.replace("[", " ")
    check_text_str = check_text_str.replace("]", " ")

    return check_text_str


def clean_date_string(string):
    string_cleaned = string
    string_cleaned = string_cleaned.replace(",", " ")
    string_cleaned = string_cleaned.replace(".", " ")
    string_cleaned = string_cleaned.replace("  ", " ")
    string_cleaned = string_cleaned.replace("  ", " ")
    string_cleaned = re.sub(clean_string_pattern_1, "", string_cleaned)

    return string_cleaned


def find_additional_date_after(string):
    res = []
    ad_hoc_matches = re.findall(add_hoc_date_pattern, string)
    if ad_hoc_matches is not None:
        # print("ad_hoc matches:", ad_hoc_matches)
        res.extend([(m[0], None) for m in ad_hoc_matches])

    ad_hoc_matches_2 = re.findall(add_hoc_date_pattern_2, string)
    if ad_hoc_matches_2 is not None:
        # print("ad_hoc matches:", ad_hoc_matches_2)
        res.extend([(m[0], None) for m in ad_hoc_matches_2])
    return res


def find_additional_date_before(string):
    res = []
    new_string = string
    ad_hoc_matches_4 = re.findall(add_hoc_date_pattern_4, string)
    if add_hoc_date_pattern_4 is not None:
        for r in ad_hoc_matches_4:
            cr = r[0]
            res.append((cr, None))
            new_string.replace(cr, "")

    ad_hoc_matches_3 = re.findall(add_hoc_date_pattern_3, new_string)
    if ad_hoc_matches_3 is not None:
        # print("ad_hoc matches:", ad_hoc_matches)
        res.extend([(m[0], None) for m in ad_hoc_matches_3])
    return res


def filter_valid_date_string(res, string, disable_clean_date):
    res_filtered = []
    new_string = string
    for r in res:
        text, _ = r

        # print("curr text:", text)
        if not any(char.isdigit() for char in text):
            if not disable_clean_date and text in weekdays:
                pass
            else:
                continue
        # TODO: this is clearly a hack, need to implement date filtering heuristic to make this principaled
        elif "1 month" in text or "3 months" in text or "1 week" in text or "18 pages" in text or "nine" in text \
                or "four" in text or "minute" in text or "#" in text or "and" in text or "H1" in text or \
                ("in" in text and "March" not in text) or "of" in text or text == "2015":
            continue
        elif re.match(digit_1_3_5, text) is not None or re.match(filter_out_pattern_1, text) is not None:
            # print("here")
            continue
        if "%" in text:
            continue

        res_filtered.append(r)
        new_string = new_string.replace(text, "")

    return res_filtered, new_string


def find_date(string, fuzzy=False, disable_clean_date=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """

    string = clean_date_string(string)
    # print("cleaned_string: {}".format(string))

    # print("string:", string)
    res_filtered = []

    r = find_additional_date_before(string)
    res, string = filter_valid_date_string(r, string, disable_clean_date)
    res_filtered.extend(res)

    if "first year" in string.lower():
        res_filtered.append(("first year", None))

    if "second year" in string.lower():
        res_filtered.append(("second year", None))

    if "third_year" in string.lower():
        res_filtered.append(("third year", None))

    try:
        r = search_dates(string, languages=['en'])
        if r is None:
            r = []
    except Exception:
        return res_filtered

    if len(r) == 0:
        res_filtered.extend(find_additional_date_after(string))
        return res_filtered

    res, new_string = filter_valid_date_string(r, string, disable_clean_date)
    res_filtered.extend(res)
    res_filtered.extend(find_additional_date_after(new_string))

    #   print(res_filtered)
    return res_filtered


def find_time(string):
    possible_times = find_date(string)
    # print("possible_times:", possible_times)
    res_filtered = []
    if len(possible_times) > 0:
        for text, _ in possible_times:
            if any([w in string for w in weekdays]) or time_pattern.match(
                    text) is not None or "am" in text or "pm" in text:
                res_filtered.append(text)

        return res_filtered
    else:
        return []


def find_location(string):
    string = re.sub(r'#\d{1,5}', ' ', string)
    results = [str(r) for r in pyap.parse(string, country='US')]
    return results


def decode_datetime(obj):
    if b'__datetime__' in obj:
        obj = datetime.datetime.strptime(obj["as_str"], "%Y%m%dT%H:%M:%S.%f")
    return obj


def encode_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return {'__datetime__': True, 'as_str': obj.strftime("%Y%m%dT%H:%M:%S.%f")}
    return obj
