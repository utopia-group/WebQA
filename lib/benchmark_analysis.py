import re
from collections import defaultdict
from typing import List, Tuple, Dict

from lib.grammar import constant
from lib.spec import ExtractExample, Task
from lib.tree import ParseTree, HTMLTree, HTMLNode, LabelTree, ListTree
from lib.utils.misc_utils import clean_context_string, split_str_pattern, find_date, find_location


class TrainingSet:
    def __init__(self, schema: Dict):
        self.schema = schema
        self.selected_set = {'both_tb': [], 'qa_tb': [], 'keyword_tb': [], 'keyword_in_leaf_disjoint_tb': [],
                             'keyword_force_tb': []}

    def add(self, category, tb) -> List[str]:
        print(self.selected_set)
        if len(self.selected_set[category]) >= self.schema[category]:
            return []

        if tb not in self.selected_set[category]:
            self.selected_set[category].append(tb)
            return [tb.name]
        else:
            raise ValueError("same benchmark already added")

    def remove(self, category, tb_names):
        tmp_list = []
        for tb in self.selected_set[category]:
            if not tb.name in tb_names:
                tmp_list.append(tb)
        self.selected_set[category] = tmp_list

    def filled(self, category):
        if len(self.selected_set[category]) >= self.schema[category]:
            return True

        return False

    def __repr__(self):
        return str(self.selected_set)


class Feature:
    def __init__(self):
        pass

    def __repr__(self):
        return str(self.__dict__)


class TrainFeature(Feature):
    def __init__(self, match_section_qa, header_keyword, gt_in_single_node, gt_in_list, gt_in_parallel,
                 keyword_or_const_str_in_gt, entities, csv_in_gt):
        super().__init__()
        self.match_section_qa: bool = match_section_qa
        self.header_keyword: bool = header_keyword
        self.gt_in_single_node: bool = gt_in_single_node
        self.gt_in_list: bool = gt_in_list
        self.gt_in_parallel: bool = gt_in_parallel
        self.keyword_or_const_str_in_gt: bool = keyword_or_const_str_in_gt
        self.entities: List = entities
        self.csv_in_gt: bool = csv_in_gt


class TestFeature(Feature):
    def __init__(self):
        super().__init__()


class TestSectionFeature(TestFeature):
    def __init__(self, match_section_qa, header_keyword, leaves_contain_const, multiple_keyword_nodes=False):
        super().__init__()
        self.match_section_qa: List = match_section_qa
        self.header_keyword: List = header_keyword
        self.leaves_contain_const: List = leaves_contain_const

        self.multiple_keyword_nodes: bool = multiple_keyword_nodes


class TestClusterFeature(TestFeature):
    def __init__(self):
        super().__init__()
        self.single_node = []
        self.parallel_leaf = []
        self.contain_list = []
        self.contain_csv = []
        self.contain_const_str = []
        self.neither = []
        self.entities: Dict[str, list] = {}

    def sort_cluster_by_name(self):
        self.single_node = sorted(self.single_node, key=lambda x: int(x.name.split('_')[1]), reverse=True)
        self.parallel_leaf = sorted(self.parallel_leaf, key=lambda x: int(x.name.split('_')[1]), reverse=True)
        self.contain_csv = sorted(self.contain_csv, key=lambda x: int(x.name.split('_')[1]), reverse=True)
        self.contain_const_str = sorted(self.contain_const_str, key=lambda x: int(x.name.split('_')[1]), reverse=True)
        self.neither = sorted(self.neither, key=lambda x: int(x.name.split('_')[1]), reverse=True)

    def get_cluster_exclude_benchmarks(self, exclude_tb_names):
        new_cluster = TestClusterFeature()
        for key, value in vars(self).items():
            if isinstance(value, list):
                for tb in value:
                    if tb.name not in exclude_tb_names:
                        vars(new_cluster)[key].append(tb)

            if isinstance(value, dict):
                for k, v in value.items():
                    if k not in exclude_tb_names:
                        vars(new_cluster)[key][k] = v

        return new_cluster

    def get_all_benchmarks(self):
        return list(set(self.single_node + self.parallel_leaf + self.contain_list + self.contain_csv +
                        self.contain_const_str + self.neither))

    def distinct_benchmark_count(self):
        return len(self.entities.keys())


class BenchmarkAnalysis:
    def __init__(self, dsl, task):
        self.task: Task = task
        self.dsl: ExtractorDSL = dsl

        self.dsl.task = task

    def generate_train_feature(self, tb) -> TrainFeature:
        curr_train_feature = TrainFeature(match_section_qa=self.match_section_qa(tb),
                                          header_keyword=self.header_keyword(tb),
                                          gt_in_single_node=self.gt_in_single_node(tb),
                                          gt_in_list=self.gt_in_list(tb),
                                          gt_in_parallel=self.gt_in_parallel(tb),
                                          keyword_or_const_str_in_gt=self.const_string_in_gt(tb),
                                          entities=self.entities_in_gt_nodes(tb),
                                          csv_in_gt=self.csv_in_gt_nodes(tb))
        return curr_train_feature

    def generate_test_section_feature(self, tb) -> TestSectionFeature:
        if tb.pt.start_node.is_content_list_tree:
            leaves_contain_const = self.const_str_in_leave_node(tb.pt.list_trees[
                                                                    tb.pt.start_node.content],
                                                                is_list_tree=True, train=False)
        else:
            leaves_contain_const = self.const_str_in_leave_node(tb.pt, train=False)

        header_keyword, multiple_answer = self.header_keyword(tb, train=False)
        curr_test_section_feature = TestSectionFeature(match_section_qa=self.match_section_qa(tb, train=False),
                                                       header_keyword=header_keyword,
                                                       leaves_contain_const=leaves_contain_const,
                                                       multiple_keyword_nodes=multiple_answer)
        return curr_test_section_feature

    def cluster_this_benchmark(self, features: TestClusterFeature, tb, subtree: ParseTree):

        print("clustering {}".format(tb.name))
        print("subtree:", subtree)
        assigned = False

        if self.single_node_section(subtree):
            assigned = True
            features.single_node.append(tb)

        if self.leaf_node_in_parallel(subtree):
            assigned = True
            features.parallel_leaf.append(tb)

        if self.subtree_contains_list(subtree):
            assigned = True
            features.contain_list.append(tb)

        if self.csv_in_subtree(subtree):
            assigned = True
            features.contain_csv.append(tb)

        if subtree.start_node.is_content_list_tree:
            if self.const_str_in_leave_node(subtree.list_trees[subtree.start_node.content], is_list_tree=True):
                assigned = True
                features.contain_const_str.append(tb)
        else:
            if self.const_str_in_leave_node(subtree):
                assigned = True
                features.contain_const_str.append(tb)

        if not assigned:
            features.neither.append(tb)

        features.entities[tb.name] = self.entities_in_subtree(subtree)
        # print(features)

    """
    enumerate some feature detection here:
    1. header-children
        -> list?
        -> not list?
        -> how to locate this header
    2. everything in the leaf node
    3. no obvious way of locating it
        -> in this case we probably don't want to include this as a training set anyway
    """

    def match_section_qa(self, tb, train=True):
        # NOTE: k=1 in this case, may change
        qa_match_id_helper_res = self.dsl.match_qa_node_id_helper(tb.pt.file_name, self.task.q, 5)
        qa_match_result = sorted(set(qa_match_id_helper_res), key=qa_match_id_helper_res.index)
        qa_output = self.dsl.qa_node_id_helper(tb.pt.file_name, self.task.q)['{}-{}'.format(tb.pt.start_node.id, 0)]
        print("qa_outputs:", qa_output)
        print("match_qa_result:", qa_match_result)

        qa_match_result = [res for res in qa_match_result if not (res == (0, 0) or res == (999999, 0))]

        if train:
            if tb.gt.gt_subtree.gt_nodes_any_correct:
                gt_nodes = list(set(tb.gt.gt_subtree.gt_nodes))
                gt_ancestors = []
                for gt_node_id in gt_nodes:
                    gt_ancestors.append(tb.pt.node_to_ancestor_mapping[gt_node_id][-1])
                if len(qa_match_result) == 1 and any(ancestor == qa_match_result[0] for ancestor in gt_ancestors):
                    return True
                else:
                    return False

            else:
                gt_closest_common_ancestor = tb.gt.gt_subtree.get_gt_nodes_common_ancestor(tb.pt)[-1]
                print("gt_closest_common_ancestor:", gt_closest_common_ancestor)
                if len(qa_match_result) == 1 and gt_closest_common_ancestor == qa_match_result[0]:
                    return True
                else:
                    return False
        else:
            return qa_match_result

    def match_keyword_helper_2(self, pt, keywords, k, threshold=0.0):
        header_candidates = pt.get_header()
        multiple_answer = False

        # filter out irrelavant keyword to increase accuracy

        # print("match_keyword_helper:", header_candidates)
        header_candidates_str = list(header_candidates.keys())
        results = self.dsl.sbert_helper(pt.file_name, keywords)
        # print("keyword:",keywords)
        results = list(results)

        if pt.file_name.startswith('clinic') and any(('service' in k.lower() or 'treatment' in k.lower()) for k in
                                                     keywords):
            new_results = []
            contain_exact_match = False
            for header_str, res in results:
                new_results_each = []
                for ref_str, score in res:
                    if score > 0.99:
                        contain_exact_match = True
                        break
                    if all(w in header_str.split() for w in ref_str.split()) and ref_str.lower() in header_str.lower():
                        new_results_each.append((ref_str, 1.0))
                    else:
                        new_results_each.append((ref_str, score))
                if contain_exact_match:
                    break
                new_results.append((header_str, new_results_each))

            if not contain_exact_match:
                # rerank
                results = list(sorted(new_results, key=lambda x: max(ref_str[1] for ref_str in x[1]), reverse=True))

        return_nodes_idx = []
        max_score = max([ref_str[1] for ref_str in results[0][1]])
        if max_score < threshold:
            return [], multiple_answer

        if k == 1:
            results_tmp = []
            # print("max_score")

            for cand_str, ref_entries in results:
                curr_max = max([ref_str[1] for ref_str in ref_entries])
                # print(cand_str, " curr_max:", curr_max)
                if curr_max == max_score:
                    results_tmp.append((cand_str, ref_entries))
                else:
                    break
            results = results_tmp
        else:
            results = [res for res in results if max([ref_str[1] for ref_str in res[1]]) >= threshold]
            results = results[:k]
        for cand_str, _ in results:
            nodes_ids_l = header_candidates[cand_str]
            if len(nodes_ids_l) > 1:
                print("multiple_answer nodes:", nodes_ids_l)
                multiple_answer = True
            return_nodes_idx.extend(nodes_ids_l)
        # print(return_nodes_idx)
        return return_nodes_idx, multiple_answer

    def header_keyword(self, tb, train=True):
        assert isinstance(tb.pt, ParseTree)

        if tb.pt.file_name.startswith('clinic'):
            keyword_threshold = 0.8
        else:
            keyword_threshold = 0.9
        match_keyword_helper_res, multiple_answer = self.match_keyword_helper_2(tb.pt, self.task.keyword, 3,
                                                                                threshold=keyword_threshold)
        keyword_match_result = sorted(set(match_keyword_helper_res), key=match_keyword_helper_res.index)
        keyword_match_result_post_processed = []
        for res in keyword_match_result:
            keyword_match_result_post_processed.append(res)
            fake_list_node_tuple = tb.pt.get_fake_list_node_tuple(res)
            if fake_list_node_tuple is not None:
                keyword_match_result_post_processed.append(fake_list_node_tuple)
        sbert_outputs = self.dsl.sbert_helper(tb.pt.file_name, self.task.keyword)
        print("sbert_outputs:", sbert_outputs)
        print("keyword_match_result:", keyword_match_result_post_processed)

        if len(keyword_match_result_post_processed) == 0:
            if train:
                return False
            else:
                return keyword_match_result_post_processed, multiple_answer

        if train:
            gt_closest_common_ancestor = tb.gt.gt_subtree.get_gt_nodes_common_ancestor(tb.pt)[-1]
            print("gt_closest_common_ancestor:", gt_closest_common_ancestor)
            if tb.pt.file_name == 'conf_2' and gt_closest_common_ancestor[0] == 13 and len(
                    keyword_match_result_post_processed) == 4:
                return True
            if tb.pt.file_name == 'class_13' and gt_closest_common_ancestor[0] == 1000 and len(
                    keyword_match_result_post_processed) == 1:
                return True
            if tb.pt.file_name == 'clinic_1' and gt_closest_common_ancestor[0] == 90000 and len(
                    keyword_match_result_post_processed) == 3:
                return True
            if tb.pt.file_name == 'clinic_1' and gt_closest_common_ancestor[0] == 235 and len(
                    keyword_match_result_post_processed) == 2:
                return True
            if gt_closest_common_ancestor in keyword_match_result_post_processed:
                return True
            else:
                return False
        else:
            return keyword_match_result_post_processed, multiple_answer

    def subtree_contain_gt_1(self, tb, section_node_id):
        gt_closest_common_ancestor = tb.gt.gt_subtree.get_gt_nodes_common_ancestor(tb.pt)[-1]
        if section_node_id == gt_closest_common_ancestor:
            return True
        else:
            return False

    def subtree_contain_gt_2(self, tb, subtree):

        def check_curr_tuple_in_subtree(gt_tree_node_id, gt_list_node_id):
            if gt_tree_node_id not in node_id_in_subtree:
                return False

            if gt_list_node_id is not None and not gt_list_node_id == 0:
                # print(gt_tree_node_id, gt_list_node_id)
                curr_list_tree = subtree.list_trees[subtree.nodes[gt_tree_node_id].content]
                node_id_in_list_subtree = [node_id for node_id in curr_list_tree.nodes.keys() if
                                           curr_list_tree.nodes_in_subtree.get(node_id) is not None]
                print("node_id_in_list_subtree {}: {}".format(tb.name, node_id_in_list_subtree))
                if gt_list_node_id not in node_id_in_list_subtree:
                    return False
            return True

        node_id_in_subtree = [node_id for node_id in subtree.nodes.keys() if subtree.nodes_in_subtree.get(node_id) is
                              not None]
        print("node_id_in_subtree {}: {}".format(tb.name, node_id_in_subtree))
        print("gt_node_id:", tb.gt.gt_subtree.refine_duplicates()[1])

        if tb.gt.gt_subtree.gt_nodes_any_correct:
            return any(check_curr_tuple_in_subtree(gt_tree_node_id, gt_list_node_id) for gt_tree_node_id,
                                                                                         gt_list_node_id in
                       tb.gt.gt_subtree.refine_duplicates()[1])
        else:
            return all(check_curr_tuple_in_subtree(gt_tree_node_id, gt_list_node_id) for gt_tree_node_id,
                                                                                         gt_list_node_id in
                       tb.gt.gt_subtree.refine_duplicates()[1])

    def const_string_in_gt(self, tb):
        # keyword_and_const_str = self.task.keyword + self.task.const_str
        keyword_and_const_str = self.task.const_str
        for gt_node_id in tb.gt.gt_subtree.gt_nodes:
            tree_node, list_node = tb.pt.get_node_by_tuple_id(gt_node_id)
            node_text = clean_context_string(tree_node.context_gen()).lower() if list_node is None else \
                clean_context_string(list_node.context_gen()).lower()

            for word in keyword_and_const_str:
                if len(word.split(' ')) > 1:
                    if word.lower() in node_text:
                        return True
                else:
                    split_content = re.split(split_str_pattern, node_text)
                    if word.lower() in [s.lower() for s in split_content]:
                        return True

        return False

    # get all leave nodes (no matter what type of tree it is)
    def leave_nodes_of_tree(self, tree: HTMLTree, is_list_tree=False) -> List[Tuple[int, int]]:
        leave_nodes = []
        curr_children_ids = tree.to_children_edges[tree.start_node.id]
        # print("curr_children_ids:", curr_children_ids)
        while len(curr_children_ids) > 0:
            new_children_ids = []
            for child_id in curr_children_ids:
                # print("child_id:", child_id)
                if tree.to_children_edges.get(child_id) is None:
                    if is_list_tree:
                        assert isinstance(tree, ListTree)
                        leave_nodes.append((tree.tree_id, child_id))
                    else:
                        assert isinstance(tree, ParseTree)
                        child_node = tree.get_node(child_id)
                        if child_node.is_content_list_tree:
                            leave_nodes.extend(self.leave_nodes_of_tree(tree.list_trees[child_node.content],
                                                                        is_list_tree=True))
                        else:
                            assert isinstance(tree, ParseTree)
                            leave_nodes.append((child_id, 0))
                else:
                    new_children_ids.extend(tree.to_children_edges[child_id])
            curr_children_ids = new_children_ids

        return leave_nodes

    def const_str_in_leave_node(self, subtree: HTMLTree, is_list_tree=False, train=True):

        # keyword_and_const_str = self.task.keyword + self.task.const_str
        keyword_and_const_str = self.task.const_str
        leave_nodes_ids = self.leave_nodes_of_tree(subtree, is_list_tree)

        # print("leave_nodes_of_tree:", leave_nodes_ids)

        leave_node_ids_contains_string = []

        # print("keyword_or_const_str_in_leave subtree:", subtree)
        # print("self.nodes:", subtree.nodes)

        for leave_node_tuple_id in leave_nodes_ids:
            # print("leave_node_tuple_id:", leave_node_tuple_id)
            if isinstance(subtree, ParseTree):
                tree_node, list_node = subtree.get_node_by_tuple_id(leave_node_tuple_id)
                leave_node = tree_node if list_node is None else list_node
            else:
                assert isinstance(subtree, ListTree)
                leave_node = subtree.get_node(leave_node_tuple_id[1])

            assert isinstance(leave_node, HTMLNode)
            node_text = clean_context_string(leave_node.context_gen()).lower()

            for word in keyword_and_const_str:
                if len(word.split(' ')) > 1:
                    if word.lower() in node_text:
                        if train:
                            return True
                        else:
                            leave_node_ids_contains_string.append(leave_node_tuple_id)
                else:
                    split_content = re.split(split_str_pattern, node_text)
                    # print("split_content:", split_content)
                    if word.lower() in [s.lower() for s in split_content]:
                        if train:
                            return True
                        else:
                            leave_node_ids_contains_string.append(leave_node_tuple_id)

        if train:
            return False
        else:
            return leave_node_ids_contains_string

    def gt_in_single_node(self, tb):
        return len(tb.gt.gt_subtree.gt_nodes) == 1

    def single_node_section(self, subtree: ParseTree):
        num_nodes = subtree.get_subtree_num_nodes()
        if num_nodes == 1 or num_nodes == 2:
            return True

        return False

    def gt_in_list(self, tb: ExtractExample):
        for gt_node_id in tb.gt.gt_subtree.gt_nodes:
            tree_node, list_node = tb.pt.get_node_by_tuple_id(gt_node_id)
            if list_node is not None:
                return True

        return False

    def subtree_contains_list(self, subtree: ParseTree):
        if subtree.start_node.is_content_list_tree:
            return True

        parent_keys = list(subtree.to_parent_edges.keys())
        parent_values = list(subtree.to_parent_edges.values())
        leaf_node_ids = [k for k in parent_keys if k not in parent_values and not k == subtree.start_node.id]
        for leaf_nid in leaf_node_ids:
            if subtree.nodes[leaf_nid].is_content_list_tree:
                return True

    # probably need to improve this
    def gt_in_parallel(self, tb: ExtractExample):
        label_tree, gt_nodes = tb.gt.gt_subtree.refine_duplicates()
        assert isinstance(label_tree, LabelTree)
        gt_nodes_parent_id = []
        for gt_node_id in gt_nodes:
            curr_node_id = label_tree.node_content_to_id[str(gt_node_id)]
            gt_nodes_parent_id.append(label_tree.get_parent_id(curr_node_id))

        if len(list(set(gt_nodes_parent_id))) == 1:
            return True
        else:
            return False

    def leaf_node_in_parallel(self, subtree: ParseTree):
        parent_keys = list(subtree.to_parent_edges.keys())
        parent_values = list(subtree.to_parent_edges.values())
        if len(parent_keys) > 0:
            leaf_node_ids = [k for k in parent_keys if k not in parent_values and not k == subtree.start_node.id]
        else:
            leaf_node_ids = [subtree.start_node.id]
        leaf_node_parent_ids = []
        for leaf_nid in leaf_node_ids:
            leaf_node = subtree.nodes[leaf_nid]
            if leaf_node.is_content_list_tree:
                curr_list_tree = subtree.list_trees[leaf_node.content]
                list_tree_parent_keys = list(curr_list_tree.to_parent_edges.keys())
                list_tree_parent_value = list(curr_list_tree.to_parent_edges.values())
                list_tree_leaf_node_ids = [k for k in list_tree_parent_keys if
                                           k not in list_tree_parent_value and not k ==
                                                                                   curr_list_tree.start_node.id]
                leaf_node_parent_ids.extend([curr_list_tree.to_parent_edges[lt_leaf_nid] for lt_leaf_nid in
                                             list_tree_leaf_node_ids])
            else:
                leaf_node_parent_ids.append(subtree.to_parent_edges[leaf_nid])

        # print("leaf_node_parent_ids:", set(leaf_node_parent_ids))

        if len(list(set(leaf_node_parent_ids))) == 1:
            return True
        return False

    def entities_of_string(self, spacy_str):
        entities = set()

        # print(gt_string)
        # print(find_date(str(gt_string),disable_clean_date=True))
        for ent_l in constant.PRED_ENTITY:
            if ent_l in [e.label_ for e in spacy_str.ents]:
                entities.add(ent_l)
            if ent_l == 'DATE' or ent_l == 'TIME':
                find_date_res = find_date(str(spacy_str), disable_clean_date=True)
                if find_date_res is not None and len(find_date_res) > 0:
                    entities.add(ent_l)
            if ent_l == 'LOC':
                find_loc_res = find_location(str(spacy_str))
                if len(find_loc_res) > 0:
                    entities.add(ent_l)
        if len(list(spacy_str.noun_chunks)) > 0:
            entities.add('NOUN')

        return list(entities)

    def context_gen_for_node_id_tuple(self, tb, tuple):
        tree_node, list_node = tb.pt.get_node_by_tuple_id(tuple)
        if list_node is not None:
            context = list_node.context_gen()
        else:
            context = tree_node.context_gen()
        return context

    def entities_in_gt_nodes(self, tb: ExtractExample):

        gt_string = self.dsl.nlp_api.spacy_init(
            '    '.join([self.context_gen_for_node_id_tuple(tb, gt_node_id) for gt_node_id in
                         tb.gt.gt_subtree.gt_nodes]))

        return self.entities_of_string(gt_string)

    def entities_in_subtree(self, subtree: ParseTree):

        context = self.dsl.nlp_api.spacy_init(subtree.context_gen(subtree.start_node))
        return self.entities_of_string(context)

    def csv_in_gt_nodes(self, tb: ExtractExample):

        # check if gt nodes are extracted from a csv structure
        if len(tb.gt.gt_subtree.gt_nodes) == 1:
            # if there is only one gt_nodes, then the number of comma should be at least #gt - 1
            context = self.context_gen_for_node_id_tuple(tb, tb.gt.gt_subtree.gt_nodes[0])
            comma_count = context.count(',')
            gt_count = len(tb.gt.gt_str.get_labels_multi_options())

            if comma_count >= gt_count - 1:
                return True
            else:
                return False

        # first split the context in a node by comma, then check if any of the csv contains a ground truth until all
        # the ground truth are included in some csv element
        gt_multiple_options = tb.gt.gt_str.get_labels_multi_options()
        gt_check = [False for _ in gt_multiple_options]

        for node_tuple_id in tb.gt.gt_subtree.gt_nodes:
            context = self.context_gen_for_node_id_tuple(tb, node_tuple_id)
            context_split = context.split(',')
            for idx, _ in enumerate(gt_check):
                for element in context_split:
                    if any(gt_option.lower() in element.lower() for gt_option in gt_multiple_options[idx]):
                        gt_check[idx] = True

        # NOTE: this criteria is a bit strict, considering having a non-strict one
        # print(gt_check)
        return all(gt_check)

    # NOTE: another strict criteria: we are only checking the cases of (const_str)..., (const_str)...
    def csv_in_subtree(self, subtree: ParseTree, is_list_tree=False):
        # check if any leave node contains a csv node
        leave_node_ids = self.leave_nodes_of_tree(subtree, is_list_tree)
        for leaf_node_tuple_id in leave_node_ids:
            tree_node, list_node = subtree.get_node_by_tuple_id(leaf_node_tuple_id)
            leaf_node = tree_node if list_node is None else list_node
            context = leaf_node.context_gen()

            total_count = 0
            for const_str in self.task.const_str:
                if len(const_str.split(" ")) > 1:
                    total_count += context.count(const_str)
                else:
                    split_context = re.split(split_str_pattern, context)
                    # print("split_content:", split_content)
                    total_count += len([s for s in split_context if s.lower() == const_str.lower()])

            if total_count > 1:
                comma_count = context.count(',')
                if comma_count >= total_count + 1:
                    return True

        return False


"""
all benchmark selection helpers
"""


class BenchmarkSelection:
    def __init__(self, selected_training_set, train_set_analysis, test_set_analysis, test_set_cluster):
        self.selected_training_set: TrainingSet = selected_training_set
        self.train_set_analysis: Dict[str, TrainFeature] = train_set_analysis
        self.test_set_analysis: Dict[str, TestSectionFeature] = test_set_analysis
        self.test_set_cluster: Dict[str, TestClusterFeature] = test_set_cluster

    """
    find entities as similar as possible here
    """

    def find_most_similar_entities(self, curr_train_tb, focused_cluster,
                                   focused_cluster_name, most_similar_feature_candidates) -> List[str]:
        candidate_to_entity_score = {}
        gt_entities = self.train_set_analysis[curr_train_tb].entities
        for candidate in most_similar_feature_candidates:
            curr_candidate_entities = focused_cluster.entities[candidate.name]

            common_entities_cnt = len(set(gt_entities).intersection(curr_candidate_entities))
            precision = common_entities_cnt / len(curr_candidate_entities)
            recall = common_entities_cnt / len(gt_entities)
            f1 = (2 * precision * recall) / (precision + recall)
            candidate_to_entity_score[candidate] = f1

        candidate_to_entity_score = sorted(candidate_to_entity_score.items(), key=lambda x: x[1], reverse=True)
        print("candidate_to_entity_score:", candidate_to_entity_score)

        # just choose the highest one (regardless how many ties there are)
        print("selected similar benchmark with highest entity score:", candidate_to_entity_score[0][0])
        return self.selected_training_set.add(focused_cluster_name, candidate_to_entity_score[0][0])

    """
    find benchmarks with features as similar as possible or with some heuristics
    """

    def find_benchmark_with_most_similar_features(self, curr_train_tb, focused_cluster,
                                                  focused_cluster_name, candidates_in_cluster):

        if not self.train_set_analysis[curr_train_tb].gt_in_single_node:
            # remove all the test set with single node property from the set
            new_candidates_in_cluster = []
            for feature_list in candidates_in_cluster:
                tmp_list = []
                for idx, element in enumerate(feature_list):
                    if element not in focused_cluster.single_node:
                        tmp_list.append(element)
                new_candidates_in_cluster.append(tmp_list)
            candidates_in_cluster = new_candidates_in_cluster

        candidate_to_same_feature_count = defaultdict(int)
        for feature_list in candidates_in_cluster:
            for element in feature_list:
                candidate_to_same_feature_count[element] += 1
        print("candidate_to_same_feature_count:", candidate_to_same_feature_count)

        candidate_to_same_feature_count = sorted(candidate_to_same_feature_count.items(), key=lambda x: x[1],
                                                 reverse=True)
        if len(candidate_to_same_feature_count) == 0 or len(candidate_to_same_feature_count[0]) == 0:
            return None
        max_feature_count = candidate_to_same_feature_count[0][1]
        most_similar_feature_candidates = [tb for tb, count in candidate_to_same_feature_count if
                                           count == max_feature_count]

        if len(most_similar_feature_candidates) == 1:
            print("selected similar benchmark:", most_similar_feature_candidates[0])
            return self.selected_training_set.add(focused_cluster_name, most_similar_feature_candidates[0])

        else:
            return self.find_most_similar_entities(curr_train_tb, focused_cluster, focused_cluster_name,
                                                   most_similar_feature_candidates)

    """
    try to select distinct schemas
    """

    def select_different_train(self, curr_train_tb, curr_train_features, focused_cluster_name, exclude_names=None) -> \
            List[str]:

        focused_cluster = self.test_set_cluster[focused_cluster_name]
        added_benchmarks = []

        if exclude_names is not None:
            focused_cluster = focused_cluster.get_cluster_exclude_benchmarks(exclude_names)

        print("focused_cluster:", focused_cluster)

        heuristic_1 = False
        heuristic_2 = (len(focused_cluster.neither) == 0)
        heuristic_3 = True

        # # make sure these are indeed keywords that cannot be found using qa
        # if focused_cluster_name == "keyword_tb":
        #     # if keyword_sec[0] == qa_sec[0], ignore this benchmark

        while not self.selected_training_set.filled(focused_cluster_name):

            if heuristic_1 and heuristic_2:
                break

            all_benchmarks_in_cluster = focused_cluster.get_all_benchmarks()

            if len(all_benchmarks_in_cluster) == 0:
                break

            # first, if most of the selected contain const str,
            if 'contain_const_str' in curr_train_features and (len(focused_cluster.contain_const_str) / len(
                    all_benchmarks_in_cluster)) > 0.8:
                if 'contain_list' in curr_train_features:
                    # get rid of those cases that contain_list
                    focused_subset = set(focused_cluster.contain_const_str).difference(focused_cluster.contain_list)
                    # just in case this ends up to be an empty set
                    if len(focused_subset) == 0:
                        focused_subset = focused_cluster.contain_const_str
                else:
                    focused_subset = focused_cluster.contain_const_str
            else:
                if 'contain_list' in curr_train_features:
                    focused_subset = set(all_benchmarks_in_cluster).difference(focused_cluster.contain_list)
                    if len(focused_subset) == 0:
                        focused_subset = all_benchmarks_in_cluster
                else:
                    focused_subset = all_benchmarks_in_cluster

            # pick the one with most shared_features:
            if not heuristic_1:
                curr_subset = set(focused_subset)
                if len(focused_cluster.parallel_leaf) > 0:
                    tmp_subset = set(focused_subset).intersection(focused_cluster.parallel_leaf)
                    if len(tmp_subset) > 0:
                        curr_subset = tmp_subset

                if len(focused_cluster.contain_csv) > 0:
                    tmp_subset = set(curr_subset).intersection(focused_cluster.contain_csv)
                    if len(tmp_subset) > 0:
                        curr_subset = tmp_subset

                if len(curr_subset) == 1:
                    added_benchmarks.extend(self.selected_training_set.add(focused_cluster_name, list(curr_subset)[0]))
                else:
                    added_benchmarks.extend(self.find_most_similar_entities(curr_train_tb, focused_cluster,
                                                                            focused_cluster_name, list(curr_subset)))

                heuristic_1 = True
                continue

            if not heuristic_2:
                if len(focused_cluster.neither) == 0:
                    added_benchmarks.extend(self.selected_training_set.add(focused_cluster_name,
                                                                           focused_cluster.neither[0]))
                else:
                    added_benchmarks.extend(self.find_most_similar_entities(curr_train_tb, focused_cluster,
                                                                            focused_cluster_name,
                                                                            focused_cluster.neither))
                heuristic_2 = True
                continue

        return added_benchmarks
