import itertools
import re

import spacy
from typing import Callable, Dict, Tuple, List

from lib.config import PRINT_MATCHSECTION_INFO, read_flag
from lib.cache import cache, cache_load, cache_store
from lib.grammar.constant import ENTITY
from lib.interpreter.context import NodeContext, StrContext, PredContext
from lib.nlp.nlp import NLPFunc
from lib.spec import Task
from lib.tree import HTMLNode, ParseTree, ListNode
from lib.utils.misc_utils import printc, get_n_gram, find_date, find_time, find_location, \
    encode_datetime, decode_datetime, filter_const_getcsv_pattern, split_str_pattern, clean_context_string

class DSL:
    def __init__(self, pt: ParseTree = None, task: Task = None):
        self.pt: ParseTree = pt
        self.task: Task = task

        self.execute_prog_str = None
        self.filter_const_getcsv_prog = False

        self.nlp_api = NLPFunc() if cache is not None else NLPFunc(disable=True)


    def GetChildren(self, ln: List[NodeContext],
                    pred_nn: Callable) -> List[NodeContext]:
        """Get the children of a list of nodes that satisfy the predicate

        Args:
            ln (List[NodeContext]): A list of node 
            pred_nn (Callable): a predicate function

        Returns:
            List[NodeContext]: a list of node that is the children of node in ln that satisfy pred_nn
        """

        ret_list = []


        def helper(parse_node, list_node):

            children = []

            if list_node is None:
                if parse_node.is_content_list_tree:
                    children.extend([
                        NodeContext(parse_node, l_n, (parse_node.id, 0))
                        for l_n in self.pt.list_trees[parse_node.content].get_children_node(
                            self.pt.list_trees[parse_node.content].start_node.id)])
                else:
                    for child in self.pt.get_children_node(parse_node.id):
                        if child.is_content_list_tree:
                            children.extend([
                                NodeContext(child, l_n, (parse_node.id, 0))
                                for l_n in self.pt.list_trees[child.content].get_children_node(
                                    self.pt.list_trees[child.content].start_node.id)])
                        else:
                            if not pred_nn.__name__ == "isStructured":
                                children.append(NodeContext(
                                    child, None, (parse_node.id, 0)))
            else:
                for child in \
                        self.pt.list_trees[parse_node.content].get_children_node(list_node.id):
                    children.append(NodeContext(
                        parse_node, child, (parse_node.id, list_node.id)))

            return children

        for elem in ln:
            ret_list.extend(helper(elem.tree_node, elem.list_node))

        return ret_list

    def GetLeaves(self, ln: List[NodeContext],
                  pred_nn: Callable = None) -> List[NodeContext]:

        """Get the leaves of a list of nodes that satisfy the predicate

        Args:
            ln (List[NodeContext]): A list of nodes 
            pred_nn (Callable): a predicate function

        Returns:
            List[NodeContext]: a list of node that is the leaves of nodes in ln that satisfy pred_nn
        """

        if pred_nn is None:
            pred_nn = self.isAny

        ret_list = []

        def helper(parse_node, list_node):

            leaves = []

            if list_node is None:
                children_nodes = self.pt.get_children_node(parse_node.id)
                if parse_node.is_content_list_tree:
                    leaves.extend([
                        NodeContext(parse_node, l_n, (parse_node.id, 0))
                        for l_n in self.pt.list_trees[parse_node.content].get_leave_node(
                            self.pt.list_trees[parse_node.content].start_node.id)])
                elif (len(children_nodes) == 1 and (children_nodes[0].id == parse_node.id * 1000) and children_nodes[
                    0].is_content_list_tree):
                    leaves.extend([
                        NodeContext(children_nodes[0], l_n, (children_nodes[0].id, 0))
                        for l_n in self.pt.list_trees[children_nodes[0].content].get_leave_node(
                            self.pt.list_trees[children_nodes[0].content].start_node.id)])

                else:
                    for leaf in self.pt.get_leave_node(parse_node.id):
                        if leaf.is_content_list_tree:
                            leaves.extend([
                                NodeContext(leaf, l_n, (parse_node.id, 0))
                                for l_n in self.pt.list_trees[leaf.content].get_children_node(
                                    self.pt.list_trees[leaf.content].start_node.id)])
                        else:
                            if not pred_nn.__name__ == "isStructured":
                                leaves.append(NodeContext(
                                    leaf, None, (parse_node.id, 0)))
            else:
                for leaf in self.pt.list_trees[parse_node.content].get_leave_node(list_node.id):
                    leaves.append(NodeContext(
                        parse_node, leaf, (parse_node.id, list_node.id)))
            return leaves

        # if of the format leave(matchDoc), find it deepest leave node
        # (i.e. consider listTree and parseTree as one tree)
        # This implementation is mainly for predicate language
        if len(ln) == 1 and ln[0].tree_node.id == self.pt.start_node.id and ln[0].list_node is None:
            # print("here")
            parse_node = ln[0].tree_node
            leaves = []

            for leaf in self.pt.get_leave_node(parse_node.id):
                if leaf.is_content_list_tree:
                    leaves.extend([
                        NodeContext(leaf, l_n, (ln[0].tree_node.id, 0)) for l_n
                        in self.pt.list_trees[leaf.content].get_leave_node(
                            self.pt.list_trees[leaf.content].start_node.id)])
                else:
                    leaves.append(NodeContext(
                        leaf, None, (ln[0].tree_node.id, 0)))
            return leaves

        for elem in ln:
            ret_list.extend(helper(elem.tree_node, elem.list_node))

        return ret_list

    def isStructured(self, node):
        """Check if the given node is a list (pred_nn function)

        Args:
            node: a node in the parse tree

        Returns:
            bool: whether this node is a list
        """

        return isinstance(node, ListNode) or node.is_content_list_tree

    def isAny(self, node):
        """Check if the given node is a any type of node (pred_nn function)

        Args:
            node: a node in the parse tree

        Returns:
            bool: always return true
        """
        return True

    def GetNode(self, np_1: Callable, _input=None,
                k: int = 1, threshold: float = 0.0) -> List[NodeContext]:
        """Get the node using the function np_1

        Args:
            np_1 (Callable): get nodes using the function np_1
            k (int): get the top-k node returned from np_1
            threshold (float): pass into np_1 to only includes nodes whose np_1 confidence is greater than threshold
        Returns:
            List[NodeContext]: a list nodes that obtain using np_1
        """

        return np_1(_input, k, threshold)

    def GetRoot(self, _input=None) -> List[NodeContext]:
        """Get the root node

        Args:
            None
        Returns:
            List[NodeContext]: a list of nodes that only contain the root node of the parse tree
        """

        return [NodeContext(
            self.pt.start_node, None, (self.pt.start_node.id, 0))]

    def matchSection1(self, _input, k, threshold,
                      everything=False) -> List[NodeContext]:
        """Get a list of node using a combination of matchQA and matchKeyword operation.  (np_1 function)
           The logic is specified as:
              if matchQA gives consistent results (i.e. located the same nodes), then return the top-k nodes that returned by matchQA
              else return the top-k nodes matched by keywords.


        Args:
            _input: cache purpose argument, ignore
            k (int): get the top-k nodes
            threshold (float): include nodes whose confidence is greater than threshold
            everything (bool): not useful, ignore
        Returns:
            List[NodeContext]: a list nodes that obtained using the procedure described above
        """

        qa_results = self.qa_node_id_helper(self.pt.file_name, self.task.q)['{}-{}'.format(self.pt.start_node.id, 0)]

        # check section consistency
        section_list = []
        for res in qa_results:
            section_list.append(res['section'])
        if len(set(section_list)) == 1:
            res = self.matchQA(_input, k, threshold)
            printc(PRINT_MATCHSECTION_INFO, "matchqa matchSection1:", res)
            return res

        return_nodes, return_node_ids = self.map_answers_to_node_helper(qa_results)
        if len(set([str(n_id) for n_id in return_node_ids])) == 1:
            res = self.matchQA(_input, k, threshold)
            printc(PRINT_MATCHSECTION_INFO, "matchqa matchSection1:", res)
            return res

        res = self.matchKeyword(_input, k, threshold, everything=everything)
        printc(PRINT_MATCHSECTION_INFO, "matchkeyword matchSection1:", res)
        return res

    def matchSection2(self, _input, k, threshold=0.85,
                      everything=False) -> List[NodeContext]:

        """Get a list of node using a combination of matchQA and matchKeyword operation.  (np_1 function)
           The logic is specified as:
              if there exists nodes whose content's similarity to the keyword > threshold, get the top-k nodes returned by matchKeyword
              else return the top-k nodes matched by question answering system


        Args:
            _input: cache purpose argument, ignore
            k (int): get the top-k nodes
            threshold (float): include nodes whose confidence is greater than threshold
            everything (bool): not useful, ignore
        Returns:
            List[NodeContext]: a list nodes that obtained using the procedure described above
        """

        # header_candidates = self.pt.get_header()
        # header_candidates_str = list(header_candidates.keys())
        keyword_results = self.sbert_helper(
            self.pt.file_name, self.task.keyword)
        # printc(PRINT_MATCHSECTION_INFO, "matchSection2 keyword:", keyword_results)
        highest_probs = [
            keyword_prob for _, keyword_prob in keyword_results[0][1]]


        if threshold == 1.0:
            threshold = 0.99

        if any(p >= threshold for p in highest_probs):
            res = self.matchKeyword(
                    self.pt.file_name, k, threshold, everything=everything)
            printc(PRINT_MATCHSECTION_INFO, "matchSection2 keyword:", res)
            return res

        else:
            res = self.matchQA(self.pt.file_name, k, threshold)
            printc(PRINT_MATCHSECTION_INFO, "matchSection2 qa:", res)
            return res

    
    def matchKeyword(self, _input,
                     k, threshold, everything=False) -> List[NodeContext]:
        """Get a list of nodes by finding those contents that has similarity to provided keyword > threshold (np_1 function)


        Args:
            _input: cache purpose argument, ignore
            k (int): get the top-k nodes
            threshold (float): include nodes whose confidence is greater than threshold
            everything (bool): not useful, ignore
        Returns:
            List[NodeContext]: a list nodes that obtained using the procedure described above
        """

        keywords = self.task.keyword

        if everything:
            raise NotImplementedError
        else:
            return_nodes_idx = self.match_keyword_helper(
            self.pt.file_name, keywords, k, threshold=threshold, partial_exact_match=read_flag('partial_exact_match'))
            return_nodes = [
                NodeContext(*(self.pt.get_node_by_tuple_id(nid)), nid)
                for nid in return_nodes_idx]

            return return_nodes

    def matchQA(self, _input, k, threshold) -> List[NodeContext]:
        """Get a list of nodes by finding those contents is recognized as answer to the question by the QA system (np_1 function)

        Args:
            _input: cache purpose argument, ignore
            k (int): get the top-k nodes
            threshold (float): include nodes whose confidence is greater than threshold
        Returns:
            List[NodeContext]: a list nodes that obtained using the procedure described above
        """

        return_nodes_idx_ans = self.match_qa_node_id_helper(
            self.pt.file_name, self.task.q, k)
        printc("matchQA:", return_nodes_idx_ans)
        return [
            NodeContext(*(self.pt.get_node_by_tuple_id(nid)), nid)
            for nid in return_nodes_idx_ans]

    def ExtractContent(self, v: List[NodeContext]) -> List[StrContext]:
        """Given a list of nodes, extract its contents to form a list of strings

        Args:
            v (List[NodeContext]): a list of nodes
        Returns:
            List[StrContext]: a list strings that includes the content of the nodes
        """

        node_str_pairs = []

        for elem in v:
            p_n = elem.tree_node
            l_n = elem.list_node
            bindex = (p_n.id, l_n.id if l_n is not None else 0)

            context = self.string_context_helper(
                elem, self.pt.file_name, bindex[0], bindex[1])

            spacy_context = self.spacy_context_doc_helper(
                context, self.pt.file_name, bindex[0], bindex[1])

            if bindex[0] == 0 and bindex[1] == 0:
                node_str_pairs.append(StrContext(
                    bindex, context, spacy_context,
                    match_section_nid=elem.match_section_nid,
                    is_whole_doc=True))
            else:
                node_str_pairs.append(StrContext(
                    bindex, context, spacy_context,
                    match_section_nid=elem.match_section_nid))

        return node_str_pairs

    def Split(self, contexts: List[StrContext], c: str = ',') -> List[StrContext]:
        """Given a list of strings, split all the strings using delimiter c

        Args:
            contexts (List[StrContext]): a list of strings
            c (str): a delimiter to split the string
        Returns:
            List[StrContext]: the results of splitting the strings in contexts
        """

        results = []

        # getCSVItem(getDoc) => getCSVItem(getLeaves(getDoc))
        if len(contexts) == 1 and contexts[0].is_whole_doc:
            leaves = self.GetLeaves(self.GetRoot(self.pt), self.isAny)
            contexts = self.ExtractContent(leaves)

        for context in contexts:
            content = context.get_str_text()

            if self.filter_const_getcsv_prog:
                total_count = 0
                for const_str in self.task.const_str:
                    if len(const_str.split(" ")) > 1:
                        total_count += content.count(const_str)
                    else:
                        split_content = re.split(split_str_pattern, content)
                        total_count += len([s for s in split_content if s.lower() == const_str.lower()])


                if total_count == 1:
                    results.append(StrContext(context.bindex, content, None,
                                              match_section_nid=context.match_section_nid, partial=True))
                    continue

            content_splits = content.split(c)
            results.extend([
                StrContext(
                    context.bindex, s, None,
                    match_section_nid=context.match_section_nid, partial=True)
                for s in content_splits])

        return results

    def Filter(self, contexts: List[StrContext], np_2: Callable) -> List[StrContext]:
        """Given a list of strings, filter the strings using function np_2

        Args:
            contexts (List[StrContext]): a list of strings
            np_2 (Callable): a function that specifies how to filter the strings
        Returns:
            List[StrContext]: a list strings that includes all strings in contexts that satisfy np_2
        """
        if len(contexts) == 1 and contexts[0].is_whole_doc:
            # split the whole-doc context by sentence and
            doc_context = contexts[0]
            contexts = []
            for sent in \
                    doc_context.get_spacy_text(self.nlp_api.spacy_init).sents:
                # print("sent:", sent)
                contexts.append(
                    StrContext(
                        doc_context.bindex, sent.text, sent,
                        match_section_nid=doc_context.match_section_nid,
                        partial=True))

        ret_contexts = []

        for context in contexts:
            if np_2(context):
                ret_contexts.append(context)
        return ret_contexts

    def GetAnswer(self, contexts: List[StrContext], q=None, k: int = 1) -> List[StrContext]:
        """Given a list of strings, obtain the answers to the question using the strings as contexts

        Args:
            contexts (List[StrContext]): a list of strings
            q (str): a question
            k (int): get top-k answers

        Returns:
            List[StrContext]: a list strings that includes the answers to the question using each string in contexts as contexts
        """

        # print("in getAnswer contexts:", contexts)

        question = self.task.q if q is None else q
        results = []

        contexts_batch_qa = []
        context_id_context_map = {}
        context_counter = itertools.count()

        # the workflow is the following:
        # for each context
        # if this conext has been cached, read directly
        # if not cached then batching them
        for context in contexts:
            context_str = context.get_str_text()
            if context_str.strip() == "":
                continue

            if context.partial:
                answers = self.get_answers_cache(self.pt.file_name, context_str, question, k)
                if answers is not None:
                    results.extend([
                        StrContext(
                            context.bindex, answer['text'], None,
                            match_section_nid=context.match_section_nid,
                            partial=True)
                        for answer in answers])
                else:
                    _id = str(next(context_counter))
                    contexts_batch_qa.append((_id, context_str))
                    context_id_context_map[_id] = context
            else:
                cache_string_key = str((context.bindex[0], context.bindex[1]))
                results.extend(self.get_answer_node_id_helper(
                    context, self.pt.file_name, cache_string_key, question, k))

        # print("in getanswer  contexts_batch_qa: ", contexts_batch_qa)

        if len(contexts_batch_qa) > 0:
            # invoke batch qa
            results.extend(self.get_answer_batch(
                self.pt.file_name, contexts_batch_qa, context_id_context_map,
                question, k))
        
        # print("GetAnswers:", results)

        return results

    def GetEntity(self, contexts: List[StrContext], LABEL: str):
        """Given a list of strings, extract those strings that has the entity LABEL
        Args:
            contexts (List[StrContext]): a list of strings
            LABEL (str): a specific entity label
        Returns:
            List[StrContext]: a list strings of the entity LABEL
        """

        ret_ents_list = []
        for context in contexts:

            cache_string_key = self.generate_cache_string_key(context)
            res = []
            if LABEL == 'DATE':
                context_str = context.get_str_text()
                date_entities = self.date_helper(
                    self.pt.file_name, context_str)
                res = [
                    StrContext(
                        context.bindex, string, None,
                        match_section_nid=context.match_section_nid,
                        partial=True)
                    for string, _ in date_entities]
            elif LABEL == 'TIME':
                context_str = context.get_str_text()
                time_entities = self.time_helper(
                    self.pt.file_name, context_str)
                res = [
                    StrContext(
                        context.bindex, string, None,
                        match_section_nid=context.match_section_nid,
                        partial=True)
                    for string in time_entities]
            elif LABEL == 'NOUN':
                res = self.spacy_entity_helper(
                    context, self.pt.file_name, LABEL, cache_string_key)
                res = [
                    StrContext(
                        context.bindex, text, None,
                        match_section_nid=context.match_section_nid,
                        partial=True)
                    for text in res]
            elif LABEL == 'PERSON':
                # NOTE: I made the assumption here that a person
                # entity should contains at least two words
                res = self.spacy_entity_helper(
                    context, self.pt.file_name, LABEL, cache_string_key)
                res = [
                    StrContext(
                        context.bindex, text, None,
                        match_section_nid=context.match_section_nid,
                        partial=True)
                    for text in res]
            elif LABEL == 'LOC':
                context_str = context.get_str_text()
                location_ret = self.location_helper(self.pt.file_name, context_str)
                res = [StrContext(context.bindex, str(r), None, match_section_nid=context.match_section_nid, partial=True) for r in location_ret]
                entities = self.spacy_entity_helper(
                    context, self.pt.file_name, 'GPE', cache_string_key)
                res_check = ' '.join(location_ret)
                # print("res_check:", res_check)
                res2 = [StrContext(context.bindex, text, None, match_section_nid=context.match_section_nid, partial=True) for text in entities 
                if len(text.split()) == 1 and ('MD' not in text  and 'M.D.' not in text and 'MPA' not in text and 'LD' not in text and 'MPH' not in text and not text == 'N') and text not in res_check]
                res = res + res2
            else:
                res = self.spacy_entity_helper(
                    context, self.pt.file_name, LABEL, cache_string_key)
                res = [
                    StrContext(
                        context.bindex, text, None,
                        match_section_nid=context.match_section_nid,
                        partial=True)
                    for text in res]
            ret_ents_list.extend(res)
        return ret_ents_list

    def GetString(self, contexts: List[StrContext],
                  keywords: List[str], threshold=1.0) -> List[StrContext]:
        """Given a list of nodes, get those substrings that is similar to keywords above certain threshold

        Args:
            contexts (List[StrContextg]): a list of strings
            keywords (List[str]): a list of keywords
            threshold (float): the threshold
        Returns:
            List[StrContext]: a list strings that is at least threshold similar to the keywords in contexts
        """
        results = []
        if threshold == 1.0:
            keywords_standardize = []
            for k in keywords:
                keywords_standardize.append(
                    [k.lower(), k.lower().replace("-", " ")])

            for context in contexts:
                content = context.get_str_text().lower()
                for k_s in keywords_standardize:
                    for ks in k_s:
                        if ks in content:
                            results.append(StrContext(
                                context.bindex, ks, None,
                                match_section_nid=context.match_section_nid,
                                partial=True))
                            break
        else:
            pass
        return results

    def isSingleton(self, v: List[NodeContext]) -> PredContext:
        """Given a list of nodes, check if there is only one node in the list

        Args:
            v (List[NodeContext]): a list of nodes
        Returns:
            PredContext (wrapper of boolean): True if there is only one node in v False otherwise
        """

        if len(v) == 1:
            return PredContext(True, [n.match_section_nid for n in v])
        return PredContext(False, [n.match_section_nid for n in v])

    def AnySat(self, v: List[NodeContext], np_2: Callable, matchSection=False) -> PredContext:
        """Given a list of nodes, check if any of the nodes satisfy np_2

        Args:
            v (List[NodeContext]): a list of nodes
            np_2 (Callable): a function that uses to filter nodes
            matchSection (bool): book-keeping argument, ignore
        Returns:
            PredContext: True if there is any nodes in v that satisfy np_2 False otherwise
        """

        if len(v) == 0:
            return PredContext(False, [])

        if np_2.__name__ == "_true":
            return PredContext(True, [n.match_section_nid for n in v])

        # if use match section and the nodecontext is the root, return false directly
        # print("matchSection:", matchSection)
        # print([n.match_section_nid for n in v])
        if all(n.match_section_nid[0] == self.pt.start_node.id and n.match_section_nid[1] == 0 for n in v) and matchSection:
            return PredContext(False, [n.match_section_nid for n in v])

        contexts = self.ExtractContent(v)
        if np_2.__name__ == "hasHeader":
            eval_res = np_2(contexts)
        else:
            eval_res = any([np_2(context) for context in contexts])
        return PredContext(eval_res, [n.match_section_nid for n in v])

    def hasHeader(self, contexts: List[NodeContext], threshold: float) -> bool:
        """Given a list of nodes, check if it contains any non-leaf nodes whose is threshold similar to keywords (np_2 funtion)

        Args:
            contexts (List[NodeContext]): a list of nodes
            threshold (float): a threshold value for keyword matching
        Returns:
            bool: True if found at least one matched node False otherwise
        """

        return len(self.matchKeyword(self.pt, 1, threshold)) > 0

    def hasString(self, context: StrContext, keywords: List[str], threshold=0.9) -> bool:
        """Given a string, check if it contains any content is threshold similar to keywords (np_2 funtion)

        Args:
            context (StrContext): a string
            keywords (List[Str]): a list of keywords
            threshold (float): a threshold value for keyword matching
        Returns:
            bool: True if found at least one matched content False otherwise
        """
        # print("keywords:", keywords)
        if threshold == 1.0:
            threshold = 0.99

        node_ids = context.bindex
        check_text_str = clean_context_string(context.get_str_text())

        # exact match
        for keyword in keywords:
            if len(keyword.split(' ')) > 1:
                if keyword.lower() in check_text_str.lower():
                    return True
            else:
                split_content = re.split(split_str_pattern, check_text_str)
                if keyword.lower() in [s.lower() for s in split_content]:
                    return True

        # print("continue")

        # soft matching, how this works:
        # First build an index for all unique words
        # Get the largest length in the keyword $len
        # For each unique word, find their pos in the context,
        # extract $len-gram cover this word
        # Check if exist a string above a threshold
        # NOTE: should be able to cache the output

        cache_string_key = self.generate_cache_string_key(context)

        similarity_res = self.sbert_string_helper(
            context, self.pt.file_name, cache_string_key, keywords)

        similarity_res_values = []
        for v in similarity_res.values():
            similarity_res_values.extend(v)

        if len(similarity_res) > 0 \
                and any(
            [sim >= threshold for (_, sim) in similarity_res_values]):
            return True
        return False

    def hasEntity(self, context: StrContext, LABEL: str) -> bool:
        """Given a string, check if it contains any contents of the entity LABEL

        Args:
            context (StrContext): a string
            LABEL (str): a entity label
        Returns:
            bool: True if found at least one matched content False otherwise
        """

        context_str = context.get_str_text()

        if LABEL == 'DATE':
            possible_date = self.date_helper(self.pt.file_name, context_str)
            return len(possible_date) > 0

        if LABEL == 'TIME':
            possible_time = self.time_helper(self.pt.file_name, context_str)
            return len(possible_time) > 0
        
        if LABEL == 'LOC':
            possible_loc = self.location_helper(self.pt.file_name, context_str)
            cache_string_key = self.generate_cache_string_key(context)
            entities = self.spacy_entity_helper(
                context, self.pt.file_name, 'GPE', cache_string_key)
            # print("res_check:", res_check)
            res2 = [text for text in entities
                    if len(text.split()) == 1 and (
                            'MD' not in text and 'M.D.' not in text and 'MPA' not in text and 'LD' not in text and
                            'MPH' not in text and not text == 'N')]
            res = possible_loc + res2
            return len(res) > 0

        cache_string_key = self.generate_cache_string_key(context)
        entities = self.spacy_entity_helper(
            context, self.pt.file_name, LABEL, cache_string_key)
        return len(entities) > 0

    def hasStrEnt(self, context: StrContext, const_str: List[str], LABEL: str, threshold=1.0) -> bool:
        """Given a string, check if it contains any contents that are similar to const_str and of the entity LABEL

        Args:
            context (StrContext): a string
            const_str (List[str]): a list of keywords
            LABEL (str): a entity label
            threshold (float): the threshold for keyword matching
        Returns:
            bool: True if found at least one matched content False otherwise
        """
        return self.hasEntity(context, LABEL) and self.hasString(context, const_str, threshold)

    def _not(self, pred: bool) -> bool:
        return not pred

    def _true(self, context: StrContext):
        return True

    """
    All the helper methods
    """

    @cache(ignore_args=[0])
    def match_keyword_helper(self, file_name, keywords, k, threshold=0.0, partial_exact_match=read_flag('partial_exact_match')):

        # print("partial_exact_match:", partial_exact_match)
        header_candidates = self.pt.get_header()

        # filter out irrelavant keyword to increase accuracy

        # print("match_keyword_helper:", header_candidates)
        header_candidates_str = list(header_candidates.keys())
        results = self.sbert_helper(self.pt.file_name, keywords)
        # print("keyword:",keywords)
        results = list(results)

        # if does not contain exact match, then find partial exact in some of the cases
        if partial_exact_match:
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
        
        
        printc(PRINT_MATCHSECTION_INFO, "matchkeyword res: ", results)
    
        return_nodes_idx = []
        max_score = max([ref_str[1] for ref_str in results[0][1]])
        if max_score < threshold:
            return []
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
            return_nodes_idx.extend(nodes_ids_l)
        # print(return_nodes_idx)
        return return_nodes_idx

    def map_answers_to_node_helper(self, answers_l) -> \
            Tuple[List[NodeContext], List[Tuple[int, int]]]:
        return_nodes = []
        return_nodes_idx = []
        for answer in answers_l:

            if answer["section"] == -1:
                continue

            ans_sec = tuple(answer["section"].split("-"))
            parse_tree_node, list_tree_node = \
                self.pt.get_node_by_tuple_id(ans_sec)

            # TODO: here we always get the parent of the section node
            if list_tree_node is None:
                if parse_tree_node.is_content_list_tree:
                    return_nodes.append(NodeContext(
                        parse_tree_node, None, (parse_tree_node.id, 0)))
                    return_nodes_idx.append((parse_tree_node.id, 0))
                elif self.pt.is_leaf(parse_tree_node.id):
                    tmp = NodeContext(
                        self.pt.get_parent(parse_tree_node.id), None,
                        (self.pt.get_parent(parse_tree_node.id).id, 0))
                    return_nodes.append(tmp)
                    return_nodes_idx.append((tmp.tree_node.id, 0))
                else:
                    # check if it has a dumb list node, if that is the case, return that dumb node instead
                    children_node = self.pt.get_children_node(parse_tree_node.id)
                    if len(children_node) == 1 and children_node[0].is_content_list_tree:
                        return_nodes.append(NodeContext(
                            children_node[0], None, (children_node[0].id, 0)
                        ))
                        return_nodes_idx.append((children_node[0].id, 0))
                    else:
                        return_nodes.append(NodeContext(
                            self.pt.get_node(parse_tree_node.id), None,
                            (self.pt.get_node(parse_tree_node.id).id, 0)))
                        return_nodes_idx.append((parse_tree_node.id, 0))
            else:
                list_sec_parent = self.pt.list_trees[parse_tree_node.content].get_parent(
                    list_tree_node.id)

                if self.pt.list_trees[parse_tree_node.content].is_leaf(list_tree_node.id):

                    # print("list_node_id:", list_tree_node.id)
                    # print("list_sec_parent_id:", list_sec_parent.id)

                    if list_sec_parent.key == "ROOT":
                        return_nodes.append(NodeContext(
                            parse_tree_node, None, (parse_tree_node.id, 0)))
                        return_nodes_idx.append((parse_tree_node.id, 0))
                    elif list_sec_parent.empty_content:
                        list_sec_parent_parent = self.pt.list_trees[parse_tree_node.content].get_parent(
                            list_sec_parent.id)
                        # print("list_sec_parent_parent_id:",
                        # list_sec_parent_parent.id)
                        if list_sec_parent_parent.key == "ROOT":
                            return_nodes.append(NodeContext(
                                parse_tree_node, None,
                                (parse_tree_node.id, 0)))
                            return_nodes_idx.append((parse_tree_node.id, 0))
                        else:
                            return_nodes.append(NodeContext(
                                parse_tree_node, list_sec_parent_parent,
                                (parse_tree_node.id, list_sec_parent_parent.id)
                            ))
                            return_nodes_idx.append(
                                (parse_tree_node.id, list_sec_parent_parent.id)
                            )
                    else:
                        return_nodes.append(NodeContext(
                            parse_tree_node, list_sec_parent,
                            (parse_tree_node.id, list_sec_parent.id)))
                        return_nodes_idx.append(
                            (parse_tree_node.id, list_sec_parent.id))
                else:
                    return_nodes.append(NodeContext(
                        parse_tree_node, list_tree_node,
                        (parse_tree_node.id, list_tree_node.id)))
                    return_nodes_idx.append(
                        (parse_tree_node.id, list_tree_node.id))
        return return_nodes, return_nodes_idx

    @cache(ignore_args=[0])
    def match_qa_node_id_helper(self, file_name, question, k):
        answers = self.qa_node_id_helper(self.pt.file_name, question)[
                      '{}-{}'.format(self.pt.start_node.id, 0)][:k]
        printc(PRINT_MATCHSECTION_INFO, "match_qa_node_id_helper answer:", answers)
        return_nodes_ans, return_nodes_idx_ans = \
            self.map_answers_to_node_helper(answers)
        return return_nodes_idx_ans

    @cache(ignore_args=[0])
    def qa_node_id_helper(self, file_name, question):
        def context_gen(title):
            contexts_l = []
            for node in [self.pt.start_node]:
                contexts_l.append(
                    "{}\n{}".format(title, self.pt.context_gen(node)))
            return contexts_l

        return self.nlp_api.run_qa_on_json(context_gen('test'), question)

    @cache(ignore_args=[0, 1])
    def string_context_helper(self, elem, file_name, pnode_id, lnode_id):
        p_n = elem.tree_node
        l_n = elem.list_node
        bindex = (p_n.id, l_n.id if l_n is not None else 0)
        if l_n is None:
            context = self.pt.print_tree_str(p_n)
        else:
            context = self.pt.list_trees[p_n.content].print_tree_str(l_n)
        context = context.replace("*", "")
        context = context.replace("&nbsp", "\t")
        return context

    @cache(ignore_args=[0, 1])
    def spacy_context_helper(self, context, file_name, pnode_id, lnode_id):
        spacy_context = self.nlp_api.spacy_init(context)
        return spacy_context.to_bytes()

    @cache(ignore_args=[0, 1], in_memory=True)
    def spacy_context_doc_helper(self, context,
                                 file_name, pnode_id, lnode_id):
        spacy_context = self.spacy_context_helper(
            context, file_name, pnode_id, lnode_id)
        vocab = self.nlp_api.get_spacy().model.vocab
        spacy_context = spacy.tokens.Doc(vocab) \
            .from_bytes(spacy_context)
        return spacy_context

    @cache(ignore_args=[0])
    def sbert_helper(self, file_name, keywords):
        header_candidates = self.pt.get_header()
        # print("sbert_helper:", header_candidates)
        header_candidates_str = list(header_candidates.keys())
        result_dict: Dict[str, List[Tuple[str, int]]] = \
            self.nlp_api.get_sbert().run(header_candidates_str, keywords)
        results = sorted(result_dict.items(),
                         key=lambda x: (max([
                             each_ref_entry[1]
                             for each_ref_entry in x[1]])),
                         reverse=True)
        return results

    @cache(ignore_args=[0, 1])
    def sbert_string_helper(self, context, file_name,
                            cache_string_key, keywords):
        similarity_res = {}
        check_text_spacy = self.get_spacy_context(context, cache_string_key)

        # TODO: the keyword to n-gram candidate should be cached
        unique_words = set()
        max_keyword_phrase_length = 0
        for keyword in keywords:
            keyword_tokenize = keyword.split(' ')
            max_keyword_phrase_length = len(keyword_tokenize) \
                if len(keyword_tokenize) > max_keyword_phrase_length \
                else max_keyword_phrase_length
            for word in keyword_tokenize:
                unique_words.add(word.lower())

        soft_match_words = set()
        for i in range(len(check_text_spacy)):
            if str(check_text_spacy[i]).lower() in unique_words:
                soft_match_words.update(get_n_gram(
                    i, check_text_spacy, max_keyword_phrase_length))

        if len(soft_match_words) > 0:
            similarity_res = self.nlp_api.get_sbert().run(
                list(soft_match_words), keywords)

        return similarity_res

    @cache(ignore_args=[0], default=encode_datetime,
           object_hook=decode_datetime)
    def date_helper(self, file_name, context_str):
        possible_date = find_date(context_str)
        if possible_date is None:
            possible_date = []
        return possible_date

    @cache(ignore_args=[0], default=encode_datetime,
           object_hook=decode_datetime)
    def time_helper(self, file_name, context_str):
        possible_time = find_time(context_str)
        if possible_time is None:
            possible_time = []
        return possible_time
    
    @cache(ignore_args=[0])
    def location_helper(self, file_name, context_str):
        possible_loc = find_location(context_str)
        
        return possible_loc

    @cache(ignore_args=[0, 1])
    def spacy_entity_helper(self, context, file_name, label, cache_string_key):

        def process_person_str(string):
            if "," in string:
                return string.split(",")[0]
           # elif "-" in string:
           #     return string.split("-")[0]
            elif "chair" in string.lower():
                return string.lower().split("chair")[0]
            else:
                return string

        content = self.get_spacy_context(
            context, cache_string_key=cache_string_key)
        if label == 'NOUN':
            ents_list = [
                n.text
                for n in content.noun_chunks]
        elif label == 'PERSON':
            ents_list = [process_person_str(e.text)
                #e.text
                for e in content.ents
                if e.label_ == label and len(e.text.split()) >= 2]
        else:
            ents_list = [
                e.text
                for e in content.ents
                if e.label_ == label]
        # print("ents_list:", ents_list)
        return ents_list

    # @cache(ignore_args=[0, 1])
    def get_answer_node_id_helper(self, context, file_name, cache_string_key,
                                  question, k):
        answers = self.qa_node_id_helper(file_name, question)["{}-{}".format(
            context.bindex[0], context.bindex[1])][:k]
        # print("answers: ", answers)
        format_res = [StrContext(
            context.bindex, answer['text'], None,
            match_section_nid=context.match_section_nid, partial=True)
            for answer in answers]
        return format_res

    def add_answers_cache(self, res, file_name, key, question, k):
        cache_store('answer_cache', file_name, key, question, k, res)

    def get_answers_cache(self, file_name, key, question, k):
        return cache_load('answer_cache', file_name, key, question, k)

    def get_answer_batch(self, file_name, context_id_strs,
                         context_id_context_map, question, k):
        # print("context_id_strs:", context_id_strs)
        answers = self.nlp_api.run_qa_on_sec_batch(
            ["{}\n[%0-0]{}[%0-0]".format(_id, context_str)
             for _id, context_str in context_id_strs], question)
        # print("answers:", answers)

        return_res = []
        for _id, context_str in context_id_strs:
            answer = answers[_id][question][:k]
            self.add_answers_cache(answer, file_name, context_str, question, k)
            context = context_id_context_map[_id]
            return_res.extend([StrContext(
                context.bindex, a['text'], None,
                match_section_nid=context.match_section_nid,
                partial=True) for a in answer])
        # print("return_res:", return_res)

        return return_res

    @cache(ignore_args=[0, 1])
    def spacy_string_helper(self, context, file_name, cache_string_key):
        check_text_spacy = context.get_spacy_text(
            self.nlp_api.spacy_init)
        return check_text_spacy.to_bytes()
        # return check_text_spacy

    # Use priority 0 to always keep in memory
    @cache(ignore_args=[0, 1], in_memory=True, priority=0)
    def spacy_string_doc_helper(self, context, file_name, cache_string_key):
        check_text_spacy = self.spacy_string_helper(
            context, file_name, cache_string_key)
        vocab = self.nlp_api.get_spacy().model.vocab
        check_text_spacy = spacy.tokens.Doc(vocab) \
            .from_bytes(check_text_spacy)
        return check_text_spacy

    def get_spacy_context(self, context, cache_string_key):
        if context.spacy_context is None:
            check_text_spacy = self.spacy_string_doc_helper(
                context, self.pt.file_name, cache_string_key)
        else:
            check_text_spacy = context.spacy_context
        return check_text_spacy

    def generate_cache_string_key(self, context):
        if context.partial:
            cache_string_key = str(context.get_str_text())
        else:
            cache_string_key = str((context.bindex[0], context.bindex[1]))
        return cache_string_key

    def set_execute_prog_str(self, prog_str, task):
        self.execute_prog_str = prog_str
        self.task = task
        self.filter_const_getcsv_prog = (re.match(filter_const_getcsv_pattern, prog_str) is not None)
        # print("prog_str:", prog_str)
        # print("filter_const_getcsv_prog:", self.filter_const_getcsv_prog)

    def unset_execute_prog_str(self):
        self.filter_const_getcsv_prog = False
        self.task = None
