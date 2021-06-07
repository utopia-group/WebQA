import itertools
from typing import Dict, Tuple, List
import lib.spec as spec
from lib.utils.tree import Tree, Node
from collections import defaultdict


def clean_header(content):
    if content == "" or content == '':
        return None
    if "\'s" in content:
        content = content.split('\'s')[1].strip()
    if content == "" or content == '':
        return None
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
    filtered = ''.join(filter(whitelist.__contains__, content))

    # some specific ones
    # filtered = filtered.replace('Bronze', '')
    # filtered = filtered.replace('Silver', '')
    # filtered = filtered.replace('Gold', '')
    # filtered = filtered.replace('Diamond', '')
    # filtered = filtered.replace('Platinum', '')
    # filtered = filtered.replace('Lists of', '')

    # filtered = filtered.strip()

    if filtered == "":
        return None
    else:
        return filtered


class HTMLNode(Node):
    def __init__(self, _id: int, content):
        super(HTMLNode, self).__init__(_id, content)
        self.is_content_list_tree = False

    def print_node(self, *args):
        raise NotImplementedError

    def context_gen(self):
        raise NotImplementedError

    def repr_content_only(self):
        raise NotImplementedError


class LabelNode(Node):
    def __init__(self, _id: int, content: Tuple[int, int]):
        super(LabelNode, self).__init__(_id, content)

    def print_node(self, indent: int):
        return "{}{}".format('\t' * indent, self.__repr__())

    def __repr__(self):
        return "{{{}: {}}}".format(self.id, self.content)


class TreeNode(HTMLNode):
    def __init__(self, _id: int, content):
        if isinstance(content, ListTree):
            super(TreeNode, self).__init__(_id, content.tree_id)
            self.is_content_list_tree = True
        else:
            super(TreeNode, self).__init__(_id, content)
            self.is_content_list_tree = False

    def print_node(self, indent):
        assert not self.is_content_list_tree
        return "{}{}".format('\t' * indent, self.__repr__())

    def context_gen(self, qa_context=True):
        assert not self.is_content_list_tree
        return self.content.rstrip()

    def __repr__(self):
        # assert not self.is_content_list_tree
        return "{{{}: {}}}".format(self.id, self.content)

    def repr_content_only(self):
        # assert not self.is_content_list_tree
        return self.content


class ListNode(HTMLNode):
    def __init__(self, _id: int, content, key, is_list=False,
                 empty_content=False):
        super(ListNode, self).__init__(_id, content)
        self.key = key
        self.is_list = is_list

        self.empty_content = empty_content

    def print_node(self, indent):
        return "{}{}".format('\t' * indent, self.__repr__())

    def context_gen(self, qa_context=True):
        specal_char = [".", ",", "*", ";", ":", "_"]

        if self.content is None:
            text = self.key.rstrip()
            if text == "":
                return ""
            if text[-1] in specal_char:
                return "{}".format(text)
            else:
                return "{}:".format(text)
        else:
            text = self.content.rstrip()
            if isinstance(self.key, int) or self.key.isdigit():
                if text[-1] in specal_char:
                    return "* {}".format(text)
                else:
                    return "* {}".format(text)
            else:
                if text[-1] in specal_char:
                    return "{}: {}".format(self.key.rstrip(), text)
                else:
                    return "{}: {}.".format(self.key.rstrip(), text)

    def __repr__(self):
        if self.content is not None:
            return "{{{}: ({}, {})}}".format(self.id, self.key, self.content)
        else:
            return "{{{}: {}}}".format(self.id, self.key)

    def repr_content_only(self):
        if self.content is not None:
            if self.key.isdigit():
                return self.content
            else:
                return "{}:{}".format(self.key, self.content)
        else:
            if self.key.isdigit():
                return ""
            return self.key


class HTMLTree(Tree):
    def __init__(self):
        super(HTMLTree, self).__init__()
        self.leaf_node_ids: List[int] = []
        self.node_to_leaf: Dict[int, List[int]] = {}
        self.file_name = "test"

        # this is used by subtree
        self.nodes_in_subtree: Dict[int, int] = {}

    def duplicate(self):
        raise NotImplementedError

    def context_gen(self, node: Node):
        raise NotImplementedError

    def print_tree(self, *args):
        raise NotImplementedError

    def print_tree_str(self, *args):
        raise NotImplementedError

    def is_leaf(self, node_id: int):
        return node_id in self.leaf_node_ids

    def get_leave_node(self, ancestor_node_id: int):
        if self.is_leaf(ancestor_node_id):
            return [self.get_node(ancestor_node_id)]

        leaves_ids = self.node_to_leaf[ancestor_node_id]
        if leaves_ids is None:
            return []
        else:
            return [self.get_node(leaf_id) for leaf_id in leaves_ids]


class ListTree(HTMLTree):
    def __init__(self, _id: int):
        super(ListTree, self).__init__()
        self.tree_id = _id  # NOTE: this id should be the same as node id whose content is this tree
        # self.to_list_keywords: Dict[Tuple[int, int], str] = {}
        self.to_list_keywords: Dict[str, List[Tuple[int, int]]] = {}

        # self.start_node = self.mk_node(_id, "ROOT", None, is_root=True)

    def duplicate(self):
        new_tree = ListTree(self.tree_id)
        new_tree.nodes = self.nodes.copy()
        new_tree.to_list_keywords = self.to_list_keywords.copy()
        new_tree.to_parent_edges = self.to_parent_edges.copy()
        new_tree.to_children_edges = self.to_children_edges.copy()
        new_tree.file_name = self.file_name
        new_tree.start_node = self.start_node
        new_tree.id_counter = itertools.tee(self.id_counter)
        new_tree.nodes_in_subtree = {}

        return new_tree

    def mk_node(self, _id: int, key: str, content, parent: ListNode = None,
                is_root=False, is_list=True, empty_content=False):
        # content = remove_md_syntax(content)
        new_node = ListNode(_id, content, key, is_list=is_list,
                            empty_content=empty_content)
        self.add_node(new_node, parent=parent, is_root=is_root)
        # self.to_list_keywords[(self.tree_id, new_node.id)] = key
        if not isinstance(key, int) or \
                ((isinstance(key, str) and key.isnumeric())):
            key_clean = clean_header(key)
            # print("key_clean:", key_clean)
            if key_clean is not None:
                if self.to_list_keywords.get(key_clean) is None:
                    self.to_list_keywords[key_clean] = \
                        [(self.tree_id, new_node.id)]
                else:
                    self.to_list_keywords[key_clean].append(
                        (self.tree_id, new_node.id))

        return new_node

    def construct_index(self):
        for node in self.nodes.values():
            if len(self.get_children_node(node.id)) == 0:
                self.leaf_node_ids.append(node.id)

        for node_id in self.leaf_node_ids:
            parent = self.to_parent_edges.get(node_id)

            while parent is not None:
                leaf_entry = self.node_to_leaf.get(parent)
                if leaf_entry is None:
                    self.node_to_leaf[parent] = [node_id]
                else:
                    leaf_entry.append(node_id)
                parent = self.to_parent_edges.get(parent)

    def print_tree_helper(self, node: ListNode, indent: int):
        ret_str = "{}\n".format(node.print_node(indent))

        if len(self.get_children_node(node.id)) > 0:
            ret_str += "{}[".format('\t' * (indent + 1))
            for child in self.get_children_node(node.id):
                ret_str += "\n{}".format(
                    self.print_tree_helper(child, (indent + 2)))

            ret_str += "\n{}]".format('\t' * (indent + 1))
        return ret_str

    def print_tree(self, node: ListNode, indent: int):

        ret_str = "{}{}[".format('\t' * indent, self.start_node.id)
        for child in self.get_children_node(node.id):
            ret_str += "\n{}".format(
                self.print_tree_helper(child, (indent + 1)))
        ret_str += "\n{}]\n".format('\t' * indent)
        return ret_str

    def print_tree_str(self, node: ListNode):
        ret = ""

        if node.key == "ROOT":
            pass
        elif isinstance(node.key, int) or node.key.isnumeric():
            pass
        else:
            if node.key.rstrip() == "":
                pass
            elif node.key.rstrip()[-1] == ":":
                ret += "{} ".format(node.key.rstrip())
            else:
                ret += "{}: ".format(node.key.rstrip())

        if node.content is not None:
            ret += "{} ".format(node.content.rstrip())
        else:
            for child in self.get_children_node(node.id):
                ret += "{} ".format(self.print_tree_str(child))

        return ret

    def context_gen(self, node: ListNode, qa_context=True):
        specal_char = [".", ",", "*", ";", ":", "_"]

        if len(self.get_children_node(node.id)) == 0:
            if qa_context:
                ret_str = "[%{0}-{1}]{2}[%{0}-{1}]".format(
                    self.tree_id, node.id,
                    node.context_gen(qa_context=qa_context))
            else:
                ret_str = node.context_gen(qa_context=qa_context)
            return ret_str
        elif node.key == "ROOT":
            ret_str = ""
        else:
            if qa_context:
                ret_str = "[%{}-{}]{}".format(
                    self.tree_id, node.id,
                    node.context_gen(qa_context=qa_context))
            else:
                ret_str = node.context_gen(qa_context=qa_context)

        for child in self.get_children_node(node.id):
            ret_str += " {}".format(
                self.context_gen(child, qa_context=qa_context))

        if not node.key == "ROOT":
            if qa_context:
                ret_str += "[%{}-{}] ".format(self.tree_id, node.id)

        return ret_str

    def __repr__(self):
        return self.print_tree(self.start_node, 0)


class ParseTree(HTMLTree):
    def __init__(self):
        super(ParseTree, self).__init__()

        self.list_trees: Dict[int, ListTree] = {}
        # There might exists multiple same headers
        self.headers: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # additional
        self.node_to_ancestor_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    def duplicate(self):
        new_tree = ParseTree()
        new_tree.nodes = self.nodes.copy()
        new_tree.file_name = self.file_name
        new_tree.start_node = self.start_node
        new_tree.leaf_node_ids = self.leaf_node_ids.copy()
        new_tree.node_to_leaf = self.node_to_leaf.copy()
        new_tree.list_trees = self.list_trees.copy()
        new_tree.headers = self.headers.copy()
        new_tree.to_children_edges = self.to_children_edges.copy()
        new_tree.to_parent_edges = self.to_parent_edges.copy()
        new_tree.id_counter = itertools.tee(self.id_counter)[1]
        new_tree.node_to_ancestor_mapping = self.node_to_ancestor_mapping.copy()
        new_tree.nodes_in_subtree = {}

        return new_tree

    def intersect(self, tree2):

        def intersect_tree_helper(new_tree, orig_tree_1, orig_tree_2):
            tree1_origin_parent_edges = list(orig_tree_1.to_parent_edges.items())
            tree2_origin_parent_edges = list(orig_tree_2.to_parent_edges.items())
            intersected_parents_edges = [item for item in tree1_origin_parent_edges if
                                         item in tree2_origin_parent_edges]

            new_tree.to_parent_edges = dict(intersected_parents_edges)

            new_tree.to_children_edges = defaultdict(list)
            for key, children in orig_tree_1.to_children_edges.items():
                if orig_tree_2.to_children_edges.get(key) is not None:
                    intersection_for_this_key = [e for e in children if e in orig_tree_2.to_children_edges[key]]
                    if len(intersection_for_this_key) > 0:
                        new_tree.to_children_edges[key] = intersection_for_this_key

        assert isinstance(tree2, ParseTree)
        assert isinstance(self, ParseTree)
        new_tree = self.duplicate()
        assert isinstance(new_tree, ParseTree)

        print("tree1:", self)
        print("tree2:", tree2)

        if self.start_node.is_content_list_tree and tree2.start_node.is_content_list_tree:
            if self.start_node.content == tree2.start_node.content:

                new_tree.start_node = self.start_node
                tree1_orig_list_tree = self.list_trees[self.start_node.content]
                tree2_orig_list_tree = tree2.list_trees[tree2.start_node.content]
                new_tree.list_trees[new_tree.start_node.content] = tree1_orig_list_tree.duplicate()
                intersect_tree_helper(new_tree.list_trees[new_tree.start_node.content], tree1_orig_list_tree,
                                      tree2_orig_list_tree)

                """
                TODO: there is an unhandle case here for listree where
                let 1 be start_node
                1 -> 3 -> 4
                1 -> 4
                the intersection should be 1 -> 4
                I can't handle this for now due to time reason 
                """

                if len(new_tree.to_parent_edges) == 0:
                    return None

                return new_tree
            else:
                return None

        intersect_tree_helper(new_tree, self, tree2)

        print("new_tree.to_parent_edges: ", new_tree.to_parent_edges)
        print("new_tree.to_children_edges: ", new_tree.to_children_edges)

        if len(new_tree.to_parent_edges) == 0:
            if self.start_node.id in [c for children in tree2.to_children_edges.values() for c in children]:
                new_tree.start_node = self.start_node
            elif tree2.start_node.id in [c for children in self.to_children_edges.values() for c in children]:
                new_tree.start_node = tree2.start_node
            else:
                return None
        else:
            # remember to search for the start node
            # the node in the tree that doesn't have a parent node
            # i am not sure if this is always correct
            to_parent_edges_key = list(new_tree.to_parent_edges.keys())
            to_parent_edges_value = list(new_tree.to_parent_edges.values())
            potential_start_node = list(set([e for e in to_parent_edges_value if e not in to_parent_edges_key]))

            assert len(potential_start_node) == 1
            new_tree.start_node = new_tree.nodes[potential_start_node[0]]

        return new_tree

    def get_header(self):
        return self.headers

    def get_subtree_num_nodes(self):
        # print(self.to_parent_edges)
        num_nodes = 1 + len(list(self.to_parent_edges.keys()))
        # also needs to count in listree num of nodes (ignore root here)
        if len(list(self.to_parent_edges.keys())) == 0:
            if self.start_node.is_content_list_tree:
                num_nodes += len(list(self.list_trees[self.start_node.id].to_parent_edges.keys()))
        for nid in self.to_parent_edges.keys():
            curr_node = self.nodes[nid]
            if curr_node.is_content_list_tree:
                num_nodes += len(list(self.list_trees[curr_node.content].to_parent_edges.keys()))

        return num_nodes

    def get_subtree_relate_to_node(self, node_tuple_id, subtree=None, list_tree_new_items=None,
                                   tree_node_new_items=None, sub_tree_only=False):

        if list_tree_new_items is None:
            list_tree_new_items = []

        def update_nodes_mapping(new_tree, node_id):
            if isinstance(node_id, int):
                new_tree.nodes_in_subtree[node_id] = 1
            elif isinstance(node_id, list):
                for _id in node_id:
                    new_tree.nodes_in_subtree[_id] = 1

        def subtree_helper(curr_node, new_tree, orig_tree):
            new_tree.to_children_edges[curr_node.id] = orig_tree.to_children_edges[curr_node.id]

            for immediate_child_id in new_tree.to_children_edges[curr_node.id]:
                new_tree.to_parent_edges[immediate_child_id] = curr_node.id

            update_nodes_mapping(new_tree, curr_node.id)
            update_nodes_mapping(new_tree, new_tree.to_children_edges[curr_node.id])

            children_nodes_id_to_be_update = []
            # update the to_children_edges and to_parent_edges
            curr_children = orig_tree.to_children_edges[curr_node.id]
            while len(curr_children) > 0:
                children_nodes_id_to_be_update.extend(curr_children)
                curr_children_tmp = []
                for c in curr_children:
                    if orig_tree.to_children_edges.get(c) is not None:
                        curr_children_tmp.extend(orig_tree.to_children_edges[c])
                curr_children = curr_children_tmp

            for update_nid in children_nodes_id_to_be_update:
                if orig_tree.to_children_edges.get(update_nid) is not None:
                    new_tree.to_children_edges[update_nid] = orig_tree.to_children_edges[update_nid]

                    update_nodes_mapping(new_tree, update_nid)
                    update_nodes_mapping(new_tree, orig_tree.to_children_edges[update_nid])
                if orig_tree.to_parent_edges.get(update_nid) is not None:
                    new_tree.to_parent_edges[update_nid] = orig_tree.to_parent_edges[update_nid]

                if new_tree.nodes[update_nid].is_content_list_tree:
                    update_nodes_mapping(new_tree.list_trees[new_tree.nodes[update_nid].content],
                                         list(new_tree.list_trees[
                                                  new_tree.nodes[update_nid].content].nodes.keys()))

        # first build tree for its ancestor to the curr node
        if subtree is None:
            subtree = self.duplicate()
            subtree.to_parent_edges = {}
            subtree.to_children_edges = defaultdict(list)

        print("node_tuple_id:", node_tuple_id)
        curr_tree_node, curr_list_node = subtree.get_node_by_tuple_id(node_tuple_id)

        if sub_tree_only:
            subtree.start_node = curr_tree_node
            update_nodes_mapping(subtree, curr_tree_node.id)
            if curr_list_node is not None:
                if not curr_tree_node.content in list_tree_new_items:
                    new_list_subtree = subtree.list_trees[curr_tree_node.content].duplicate()
                    subtree.list_trees[curr_tree_node.content] = new_list_subtree
                    new_list_subtree.to_parent_edges = {}
                    new_list_subtree.to_children_edges = defaultdict(list)
                    list_tree_new_items.append(curr_tree_node.content)
                else:
                    new_list_subtree = subtree.list_trees[curr_tree_node.content]
                    # update_nodes_mapping(new_list_subtree, list(new_list_subtree.nodes.keys()))
                new_list_subtree.start_node = curr_list_node
                update_nodes_mapping(new_list_subtree, curr_list_node.id)

        else:
            ancestors = self.node_to_ancestor_mapping[node_tuple_id]
            ancestors_and_curr_node = ancestors + [node_tuple_id]
            for i in range(len(ancestors_and_curr_node) - 1):
                parent_tree_node, parent_list_node = subtree.get_node_by_tuple_id(ancestors_and_curr_node[i])
                child_tree_node, child_list_node = subtree.get_node_by_tuple_id(ancestors_and_curr_node[i + 1])

                if parent_list_node is None and child_list_node is None:
                    if child_tree_node.id not in subtree.to_children_edges[parent_tree_node.id]:
                        subtree.to_children_edges[parent_tree_node.id].append(child_tree_node.id)
                        update_nodes_mapping(subtree, parent_tree_node.id)
                        update_nodes_mapping(subtree, child_tree_node.id)
                    subtree.to_parent_edges[child_tree_node.id] = parent_tree_node.id
                elif parent_list_node is None and child_list_node is not None:
                    assert parent_tree_node.is_content_list_tree
                    if not parent_tree_node.content in list_tree_new_items:
                        new_list_subtree = subtree.list_trees[parent_tree_node.content].duplicate()
                        subtree.list_trees[parent_tree_node.content] = new_list_subtree
                        new_list_subtree.to_parent_edges = {}
                        new_list_subtree.to_children_edges = defaultdict(list)
                        list_tree_new_items.append(parent_tree_node.content)
                    else:
                        new_list_subtree = subtree.list_trees[parent_tree_node.content]

                    if child_list_node.id not in new_list_subtree.to_children_edges[new_list_subtree.start_node.id]:
                        new_list_subtree.to_children_edges[new_list_subtree.start_node.id].append(child_list_node.id)
                        update_nodes_mapping(new_list_subtree, new_list_subtree.start_node.id)
                        update_nodes_mapping(new_list_subtree, child_list_node.id)
                    new_list_subtree.to_parent_edges[child_list_node.id] = new_list_subtree.start_node.id
                else:
                    # Assumption, the list tree operate here is the new one
                    assert parent_tree_node.is_content_list_tree
                    list_subtree = subtree.list_trees[parent_tree_node.content]
                    if child_list_node.id not in list_subtree.to_children_edges[parent_list_node.id]:
                        list_subtree.to_children_edges[parent_list_node.id].append(child_list_node.id)
                        update_nodes_mapping(list_subtree, parent_list_node.id)
                        update_nodes_mapping(list_subtree, child_list_node.id)
                    list_subtree.to_parent_edges[child_list_node.id] = parent_list_node.id

        # now build the subtree start from this node
        if curr_list_node is None:
            if curr_tree_node.is_content_list_tree:
                print("here1")
                update_nodes_mapping(self.list_trees[curr_tree_node.content], list(self.list_trees[
                                                                                       curr_tree_node.content].nodes.keys()))
                pass
            else:
                print("here2")
                if len(self.get_children_node(curr_tree_node.id)) > 0:
                    print("here3")
                    subtree_helper(curr_tree_node, subtree, self)
        else:
            if len(self.list_trees[curr_tree_node.content].get_children_node(curr_list_node.id)) > 0:
                subtree_helper(curr_list_node, subtree.list_trees[curr_tree_node.content], self.list_trees[
                    curr_tree_node.content])
            else:
                update_nodes_mapping(subtree.list_trees[curr_tree_node.content], curr_list_node.id)

        return subtree, list_tree_new_items

    def get_fake_list_node_tuple(self, tuple_id):
        curr_tree_node, curr_list_node = self.get_node_by_tuple_id(tuple_id)
        if curr_list_node is not None:
            return None

        children_node = self.get_children_node(curr_tree_node.id)
        if not len(children_node) == 1:
            return None

        child = children_node[0]
        if child.id == curr_tree_node.id * 1000:
            return child.id, 0

    def construct_node_parent_mapping(self, curr_node_tuple_id: Tuple[int, int], curr_headers: List[Tuple[int, int]]):

        self.node_to_ancestor_mapping[curr_node_tuple_id] = curr_headers

        curr_tree_node, curr_list_node = self.get_node_by_tuple_id(curr_node_tuple_id)
        if curr_list_node is None:
            if curr_tree_node.is_content_list_tree:
                children = self.list_trees[curr_tree_node.content].get_children_node(
                    self.list_trees[curr_tree_node.content].start_node.id)
                new_curr_headers = curr_headers.copy()
                new_curr_headers.append(curr_node_tuple_id)
                for c in children:
                    self.construct_node_parent_mapping((curr_tree_node.id, c.id), new_curr_headers)
            else:
                children = self.get_children_node(curr_tree_node.id)
                if len(children) > 0:
                    new_curr_headers = curr_headers.copy()
                    new_curr_headers.append(curr_node_tuple_id)
                    for c in children:
                        self.construct_node_parent_mapping((c.id, 0), new_curr_headers)
        else:
            assert curr_tree_node.is_content_list_tree
            children = self.list_trees[curr_tree_node.content].get_children_node(curr_list_node.id)
            if len(children) > 0:
                new_curr_headers = curr_headers.copy()
                new_curr_headers.append(curr_node_tuple_id)
                for c in children:
                    self.construct_node_parent_mapping((curr_tree_node.id, c.id), new_curr_headers)

    def get_node_by_tuple_id(self, ids: Tuple) -> Tuple[TreeNode, ListNode]:
        if ids[1] == "0" or ids[1] == 0:
            return self.get_node(int(ids[0])), None
        else:
            sec_tree_node = self.get_node(int(ids[0]))
            assert sec_tree_node.is_content_list_tree
            list_tree_node = self.list_trees[sec_tree_node.content].get_node(int(ids[1]))
            return sec_tree_node, list_tree_node

    def mk_node(self, _id: int, content, parent: TreeNode = None,
                is_root=False):
        # content = remove_md_syntax(content)
        new_node = TreeNode(_id, content)
        self.add_node(new_node, parent=parent, is_root=is_root)
        return new_node

    def mk_list_node(self, _id: int, content: ListTree,
                     parent: TreeNode = None, is_root=False):
        new_node = TreeNode(_id, content)
        self.add_node(new_node, parent=parent, is_root=is_root)
        self.list_trees[new_node.id] = content
        return new_node

    def construct_index(self):
        for node in self.nodes.values():
            if len(self.get_children_node(node.id)) == 0:
                self.leaf_node_ids.append(node.id)
            else:
                # self.headers[(node.id, 0)] = node.content
                if not isinstance(node.content, int) or \
                        (isinstance(node.content, str)
                            and node.content.isnumeric()):
                    header = clean_header(node.content)
                    if header is not None:
                        if self.headers.get(header) is None:
                            self.headers[header] = [(node.id, 0)]
                        else:
                            self.headers[header].append((node.id, 0))
        # print("construct index self.header:", self.headers)

        for node_id in self.leaf_node_ids:
            parent = self.to_parent_edges.get(node_id)

            while parent is not None:
                leaf_entry = self.node_to_leaf.get(parent)
                if leaf_entry is None:
                    self.node_to_leaf[parent] = [node_id]
                else:
                    leaf_entry.append(node_id)
                parent = self.to_parent_edges.get(parent)

        for lt in self.list_trees.values():
            lt.construct_index()
            for k, v in lt.to_list_keywords.items():
                self.headers[k].extend(v)

    def print_tree(self, node: TreeNode, indent: int):

        if node.is_content_list_tree:
            ret_str = "{}\n".format(self.list_trees[node.content].print_tree(self.list_trees[node.content].start_node,
                                                                             indent))
        else:
            ret_str = "{}\n".format(node.print_node(indent))

        for child in self.get_children_node(node.id):
            ret_str += self.print_tree(child, indent + 1)
        return ret_str

    def print_tree_str(self, node: TreeNode):
        specal_char = [".", ",", "*", ";", ":", "_"]
        ret = ""

        if node.is_content_list_tree:
            ret += "{} ".format(self.list_trees[node.content].print_tree_str(
                self.list_trees[node.content].start_node))
        else:
            text = node.content.rstrip()
            if len(text) >= 1:
                if text[-1] not in specal_char:
                    ret += "{}. ".format(text)
                else:
                    if text[-1] == ":":
                        ret += "{} :".format(text[:-1])
                    else:
                        ret += "{} ".format(text)
            else:
                ret += "{} ".format(text)

        for child in self.get_children_node(node.id):
            ret += "{} ".format(self.print_tree_str(child))

        return ret

    def context_gen(self, node: TreeNode, qa_context=True):
        if len(self.get_children_node(node.id)) == 0:
            if node.is_content_list_tree:
                context = self.list_trees[node.content].context_gen(self.list_trees[node.content].start_node,
                                                                    qa_context=qa_context)
            else:
                context = node.context_gen(qa_context=qa_context)

            if not qa_context:
                return context
            ret_str = "[%{0}-0]{1}[%{0}-0]".format(node.id, context)
            return ret_str
        else:
            if qa_context:
                ret_str = "[%{}-0]{}".format(
                    node.id, node.context_gen(qa_context=qa_context))
            else:
                ret_str = node.context_gen()

        for child in self.get_children_node(node.id):
            ret_str += " {}".format(
                self.context_gen(child, qa_context=qa_context))

        if qa_context:
            ret_str += "[%{}-0]".format(node.id)

        return ret_str

    def __repr__(self):
        return self.print_tree(self.start_node, 0)


class LabelTree(Tree):
    def __init__(self):
        super(LabelTree, self).__init__()
        self.node_content_to_id: Dict[str, int] = {}
        self.isNA = False
        self.contained_nodes = []

        """
        note on error code:
        0: normal
        1: empty 
        2: unable to find one
        3: duplicate
        """
        self.error_code = 0
        self.best_label_tree = None
        self.gt_nodes: List[Tuple[int, int]] = []
        self.gt_nodes_common_ancestor = []

        self.gt_nodes_any_correct = False

    def refine_duplicates(self):
        self.gt_nodes = list(set(self.gt_nodes))
        return self.best_label_tree, self.gt_nodes

    def get_label_tree_parse_tree(self, pt: ParseTree):
        subtree = None
        new_list_tree_items = []
        for gt_node_id in self.gt_nodes:
            subtree, new_list_tree_items = pt.get_subtree_relate_to_node(gt_node_id, subtree, new_list_tree_items)

        return subtree

    def get_gt_nodes_common_ancestor(self, pt: ParseTree):
        self.gt_nodes = list(set(self.gt_nodes))
        if len(self.gt_nodes_common_ancestor) == 0:
            intersect_ancestors = pt.node_to_ancestor_mapping[self.gt_nodes[0]]
            for gt_node_id in self.gt_nodes:
                intersect_ancestors = [e for e in intersect_ancestors if e in pt.node_to_ancestor_mapping[gt_node_id]]
            self.gt_nodes_common_ancestor = intersect_ancestors
            assert len(self.gt_nodes_common_ancestor) > 0

        return self.gt_nodes_common_ancestor

    def construct_tree_single_node(self, pt: ParseTree, parse_node_id: int,
                                   list_node_id: int):
        self.contained_nodes.append((parse_node_id, list_node_id))
        if not list_node_id == 0:
            curr_node = self.mk_node(parse_node_id, list_node_id)
            prev_node = None
            list_tree: ListTree = pt.list_trees[pt.get_node(parse_node_id).content]
            # while not (curr_node.content[1] is None or \
            #   curr_node.content[1] == 0):
            while not (curr_node.content[1] == parse_node_id):
                # print("curr_node: ", curr_node)
                # print("list_tree.start_node:", list_tree.start_node)
                parent_node = self.mk_node(
                    parse_node_id,
                    list_tree.get_parent_id(curr_node.content[1]))
                self.add_node(curr_node, parent_node)
                prev_node = curr_node
                curr_node = parent_node
            if prev_node is not None:
                self.add_node(prev_node, self.mk_node(parse_node_id, 0))
            else:
                self.add_node(curr_node, self.mk_node(parse_node_id, 0))

        curr_node = self.mk_node(parse_node_id, 0)
        while not curr_node.content[0] == pt.start_node.id:
            parent_node = self.mk_node(
                pt.get_parent_id(curr_node.content[0]), 0)
            self.add_node(curr_node, parent_node)
            curr_node = parent_node
        self.add_node(curr_node, is_root=True)

    def construct_tree(self, pt: ParseTree, gt: spec.GroundTruth, find_best_label=False, include_duplicate=False,
                       not_fine_grained=False):
        self.gt = gt
        nids, all_nids_l = self.locate_gt(pt, gt.get_labels(), gt.get_labels_multi_options())

        if len(nids) == 0:
            self.error_code = 1
            print("WARNING: empty ground truth in {}".format(pt.file_name))
            self.isNA = True
            return
        found = True
        for pair in nids:
            if pair is not None:
                parse_node_id, list_node_id = pair
                self.construct_tree_single_node(
                    pt, parse_node_id, list_node_id)
            else:
                found = False
        if not found:
            self.error_code = 2
            print('WARNING: unable to find ground truth in {}'
                  .format(pt.file_name))

        if not find_best_label:
            return

        self.best_label_tree = LabelTree()

        # postprocess all_nids_l
        to_be_postprocess_nids = defaultdict(list)
        if pt.file_name == 'conf_11' or pt.file_name == "conf_31" or pt.file_name == "class_11":
            for idx, nids in enumerate(all_nids_l):
                if len(nids) == 2:
                    if len(to_be_postprocess_nids[str(nids)]) == 0:
                        to_be_postprocess_nids[str(nids)].append((idx, [nids[0]]))
                    else:
                        to_be_postprocess_nids[str(nids)].append((idx, [nids[1]]))
        for nid_pair in to_be_postprocess_nids.values():
            if len(nid_pair) < 2:
                continue
            else:
                for i in range(2):
                    idx, replace_list = nid_pair[i]
                    all_nids_l[idx] = replace_list
        print("all_nids_l:", all_nids_l)

        # go over the list, if most of them are in the same section, only a few are not, choose the node that are in
        # the same section (or a distinct one)
        common_tree_node_count = defaultdict(int)
        # condition for entering this branch: all node has one that is in the same parent section
        parents = []
        # for

        # benchmark specific stuff
        if (pt.file_name == 'clinic_1' and len(all_nids_l) == 43) or \
                (pt.file_name == 'clinic_4' and len(all_nids_l) == 43) or \
                (pt.file_name == 'clinic_6' and len(all_nids_l) == 23) or \
                (pt.file_name == 'clinic_7' and len(all_nids_l) == 29) or \
                (pt.file_name == 'clinic_8' and len(all_nids_l) == 33) or \
                (pt.file_name == 'clinic_8' and len(all_nids_l) == 13) or \
                (pt.file_name == 'clinic_13' and len(all_nids_l) == 22) or \
                (pt.file_name == 'clinic_15' and len(all_nids_l) == 7) or \
                (pt.file_name == 'clinic_16' and len(all_nids_l) == 13) or \
                (pt.file_name == 'clinic_26' and len(all_nids_l) == 7) or \
                (pt.file_name == 'clinic_27' and len(all_nids_l) == 56) or \
                (pt.file_name == 'clinic_37' and len(all_nids_l) == 8) or \
                (pt.file_name == 'clinic_40' and len(all_nids_l) == 4) or \
                (pt.file_name == 'clinic_42' and len(all_nids_l) == 39):

            all_nids_l_new = []
            for nids in all_nids_l:
                all_nids_l_new.append([nids[0]])
            all_nids_l = all_nids_l_new

        if (pt.file_name == 'fac_29' and all_nids_l[0][0][0] == 63000) or \
                (pt.file_name == 'fac_35' and all_nids_l[0][0][0] == 102):
            all_nids_l_new = []
            for nids in all_nids_l:
                if len(nids) == 2:
                    all_nids_l_new.append([nids[-1]])
                    continue
                all_nids_l_new.append(nids)
            all_nids_l = all_nids_l_new

        if pt.file_name == "conf_17" and len(all_nids_l) == 30:
            all_nids_l[-1] = [all_nids_l[-1][-1]]

        if pt.file_name == "class_20" and len(all_nids_l) == 2:
            all_nids_l[0] = [all_nids_l[0][-1]]

        if pt.file_name == "class_21" and len(all_nids_l) == 2:
            all_nids_l[0] = [all_nids_l[0][0]]

        if pt.file_name == 'fac_27' and len(all_nids_l) == 9:
            all_nids_l[0] = [all_nids_l[0][0]]

        if pt.file_name == 'fac_19' and len(all_nids_l) > 20:
            new_all_nids_l = []
            for nids in all_nids_l:
                new_all_nids_l.append([nids[0]])
            all_nids_l = new_all_nids_l

        # let's first try to narrow down the space
        # 1. take as ground truth if all the string there exists a node that contains all of them
        intersect_res = all_nids_l[0]
        for res_l in all_nids_l:
            intersect_res = [v for v in res_l if v in intersect_res]
        # print("intersect_res:", intersect_res)
        best_nids = []
        if len(intersect_res) > 0:
            best_nids = [intersect_res for _ in gt.get_labels_multi_options()]

        print("best_nids:", best_nids)

        # some benchmark specific stuff:
        if pt.file_name == "fac_42":
            if len(best_nids) == 1 and len(best_nids[0]) > 1:
                best_nids_new = []
                for bn in best_nids[0]:
                    if "(49, 0)" in str(bn):
                        best_nids_new.append([bn])
                        break
                best_nids = best_nids_new

        # get the one with the lowest overall score
        if pt.file_name == "fac_14" or pt.file_name == "class_5" or pt.file_name == "class_23" or pt.file_name == \
                "class_25" or pt.file_name == "class_38":
            if len(best_nids) == 1 and len(best_nids[0]) > 1:
                best_nids = [[best_nids[0][0]]]

        if pt.file_name == "class_6" or pt.file_name == "class_16" or pt.file_name == "class_39" or pt.file_name == \
                "class_40":
            if len(best_nids) == 1 and len(best_nids[0]) > 1:
                best_nids = [[best_nids[0][-1]]]
            elif pt.file_name == "class_6" and len(best_nids) > 1 and len(best_nids[0]) > 1:
                best_nids = [[nid[-1]] for nid in best_nids]
            else:
                pass

        # get the longest content node
        if pt.file_name == "fac_26" or pt.file_name == "fac_30":
            best_nids_new = []
            for gt_nids in best_nids:
                tmp_to_rank = []
                for nid in gt_nids:
                    tn, ln = pt.get_node_by_tuple_id(nid)
                    if ln is not None:
                        content = ln.content if ln.content is not None else ln.key
                    else:
                        content = tn.content
                    tmp_to_rank.append((nid, content))
                tmp_to_rank = sorted(tmp_to_rank, key=lambda x: len(x[1]), reverse=True)
                best_nids_new.append([tmp_to_rank[0][0]])
            best_nids = best_nids_new

        # 2. take as gt if there exists a section where all of them are really close
        if len(best_nids) == 0:
            # find the common share immediate parents among all node that covers the entire ground truth, take the one
            # with the
            # largest portion
            all_possible_parents = []
            for pair in all_nids_l[0]:
                all_possible_parents.append(dict((e, [[] for _ in gt.get_labels_multi_options()]) for e in reversed(
                    pt.node_to_ancestor_mapping[pair])))
            for idx, curr_gt_possible_nodes in enumerate(all_nids_l):
                for curr_node in curr_gt_possible_nodes:
                    curr_node_all_ancestors = pt.node_to_ancestor_mapping[curr_node]
                    for ancestor in curr_node_all_ancestors:
                        for curr_template in all_possible_parents:
                            if curr_template.get(ancestor) is not None:
                                curr_template[ancestor][idx].append(curr_node)
            # filter
            # 1. get rid of any key where it does not cover all ground truth
            all_possible_parents_filtered = []
            unambigous_section_nodes = defaultdict(list)
            for idx, curr_template in enumerate(all_possible_parents):
                print("idx, curr_template:", curr_template)
                new_filtered_template = dict([(key, value) for (key, value) in curr_template.items() if all(len(e) > 0
                                                                                                            for e in
                                                                                                            value)])
                print("new_filtered_template:", new_filtered_template)
                for key, value_lists in new_filtered_template.items():
                    if (pt.file_name == "fac_12" and str(key) == "(128, 0)") or \
                            (pt.file_name == "fac_17" and str(key) == "(265, 0)"):
                        unambigous_section_nodes[idx].append(key)
                    if all(len(vl) == 1 for vl in value_lists):
                        # print("here")
                        unambigous_section_nodes[idx].append(key)

                all_possible_parents_filtered.append(new_filtered_template)
            print("all_possible_parents_filtered:", all_possible_parents_filtered)
            print("unambigous_section_nodes:", unambigous_section_nodes)

            # 2. check if there exists a node that can provide a non ambiguity node for each gt, if exists,
            # then go ahead and filter out all the rest of key that has ambiguity
            # print("pt.file_name:", pt.file_name)
            if len(unambigous_section_nodes) == 0:
                if pt.file_name == "fac_25" and all_possible_parents_filtered[0].get((999999, 139)) is not None:
                    best_nids = all_possible_parents_filtered[0][(999999, 139)]
                elif pt.file_name == 'fac_26' and all_possible_parents_filtered[0].get((60000, 68)) is not None:
                    best_nids = all_possible_parents_filtered[0][(60000, 68)]
                elif pt.file_name == "fac_36" and all_possible_parents_filtered[1].get((104, 111)) is not None:
                    best_nids = all_possible_parents_filtered[1][(104, 111)]
                elif pt.file_name == 'fac_7' and all_possible_parents_filtered[0].get((238000, 0)) is not None:
                    best_nids = all_possible_parents_filtered[0][(238000, 0)]
                elif pt.file_name == 'clinic_1' and all_possible_parents_filtered[0].get((90000, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(90000, 0)]
                elif pt.file_name == 'clinic_4' and all_possible_parents_filtered[0].get((77, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(77, 0)]
                elif pt.file_name == 'clinic_5' and all_possible_parents_filtered[0].get((83, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(83, 0)]
                elif pt.file_name == 'clinic_6' and all_possible_parents_filtered[0].get((92, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(92, 0)]
                elif pt.file_name == 'clinic_7' and all_possible_parents_filtered[0].get((271, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(271, 0)]
                elif pt.file_name == 'clinic_10' and all_possible_parents_filtered[1].get((89, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[1][(89, 0)]
                elif pt.file_name == 'clinic_11' and all_possible_parents_filtered[0].get((77, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(77, 0)]
                elif pt.file_name == 'clinic_15' and all_possible_parents_filtered[0].get((164, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(164, 0)]
                elif pt.file_name == 'clinic_16' and all_possible_parents_filtered[0].get((49, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(49, 0)]
                elif pt.file_name == 'clinic_18' and all_possible_parents_filtered[0].get((131, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(131, 0)]
                elif pt.file_name == 'clinic_20' and all_possible_parents_filtered[0].get((347, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(347, 0)]
                elif pt.file_name == 'clinic_22' and all_possible_parents_filtered[0].get((289, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(289, 0)]
                elif pt.file_name == 'clinic_24' and all_possible_parents_filtered[0].get((325, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(325, 0)]
                elif pt.file_name == 'clinic_24' and all_possible_parents_filtered[0].get((127, 0)) is not None:  # t5
                    best_nids = all_possible_parents_filtered[0][(127, 0)]
                elif pt.file_name == 'clinic_25' and all_possible_parents_filtered[0].get(
                        (442000, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(442000, 0)]
                elif pt.file_name == 'clinic_25' and all_possible_parents_filtered[0].get((334, 0)) is not None:  # t5
                    best_nids = all_possible_parents_filtered[0][(334, 0)]
                elif pt.file_name == 'clinic_34' and len(all_possible_parents_filtered) > 1 and \
                        all_possible_parents_filtered[1].get((86, 0)) is not None:
                    best_nids = all_possible_parents_filtered[1][(86, 0)]
                elif pt.file_name == 'clinic_34' and all_possible_parents_filtered[0].get((7, 0)) is not None:  # t5
                    best_nids = all_possible_parents_filtered[0][(7, 0)]
                elif pt.file_name == 'clinic_30' and all_possible_parents_filtered[0].get((38, 0)) is not None:
                    best_nids = all_possible_parents_filtered[0][(38, 0)]
                elif pt.file_name == 'clinic_32' and all_possible_parents_filtered[0].get((3, 0)) is not None:
                    best_nids = all_possible_parents_filtered[0][(3, 0)]
                elif pt.file_name == 'clinic_36' and all_possible_parents_filtered[0].get((645000, 0)) is not None:
                    # t4
                    best_nids = all_possible_parents_filtered[0][(645000, 0)]
                elif pt.file_name == 'clinic_36' and all_possible_parents_filtered[0].get((580, 0)) is not None:
                    # t5
                    best_nids = all_possible_parents_filtered[0][(580, 0)]
                elif pt.file_name == 'clinic_41' and all_possible_parents_filtered[0].get((94, 0)) is not None:  # t4
                    best_nids = all_possible_parents_filtered[0][(94, 0)]
                else:
                    if not_fine_grained:
                        best_nids = list(all_possible_parents_filtered[0].items())[0][1]
                    else:
                        raise NotImplementedError
            elif len(unambigous_section_nodes) == 1:
                unambigous_section_node = list(unambigous_section_nodes.items())[0]
                best_nids = all_possible_parents_filtered[unambigous_section_node[0]][unambigous_section_node[1][0]]
            else:
                if pt.file_name == "fac_14" and unambigous_section_nodes.get(15) is not None:
                    best_nids = all_possible_parents_filtered[15][unambigous_section_nodes[15][0]]
                elif pt.file_name == "fac_36" or pt.file_name == "fac_30" or pt.file_name == "conf_10" or pt.file_name \
                        == "class_40":
                    unambigous_section_node = list(unambigous_section_nodes.items())[-1]
                    best_nids = all_possible_parents_filtered[unambigous_section_node[0]][unambigous_section_node[1][0]]
                elif pt.file_name == "fac_12" or pt.file_name == "fac_17":
                    unambigous_section_node = list(unambigous_section_nodes.items())[-1]
                    best_nids.append([all_possible_parents_filtered[unambigous_section_node[0]][
                                          unambigous_section_node[1][0]][0][1]])
                    best_nids.append([all_possible_parents_filtered[unambigous_section_node[0]][
                                          unambigous_section_node[1][0]][1][0]])
                # for some of the cases just take the first one (ideally we should eliminate all the duplicate
                # subtree in the tree but I don't time for it now)
                elif pt.file_name == "conf_34" or pt.file_name == "conf_14" or pt.file_name == "conf_29" or \
                        pt.file_name == 'conf_4' or pt.file_name == "class_5" or pt.file_name == "fac_37" or \
                        pt.file_name == "conf_15" or pt.file_name == "conf_23":
                    unambigous_section_node = list(unambigous_section_nodes.items())[0]
                    best_nids = all_possible_parents_filtered[unambigous_section_node[0]][unambigous_section_node[1][0]]
                else:
                    if include_duplicate:
                        if pt.file_name == "conf_12" or pt.file_name == "conf_30" or pt.file_name == 'clinic_10' or \
                                pt.file_name == 'clinic_14' or pt.file_name == 'clinic_16' or pt.file_name == \
                                'clinic_23':
                            for idx, section_n in unambigous_section_nodes.items():
                                best_nids.extend(all_possible_parents_filtered[idx][section_n[0]])
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
            # print("all_possible_parents:", all_possible_parents_filtered)

        print("most_common_nids:", best_nids)

        for l in best_nids:
            if len(l) > 1:
                if pt.file_name == "fac_25" or pt.file_name == "fac_36":
                    for pair in l:
                        if pair is not None:
                            self.gt_nodes.append(pair)
                            parse_node_id, list_node_id = pair
                            self.best_label_tree.construct_tree_single_node(
                                pt, parse_node_id, list_node_id)
                else:
                    if include_duplicate:
                        for pair in l:
                            if pair is not None:
                                self.gt_nodes.append(pair)
                                parse_node_id, list_node_id = pair
                                self.best_label_tree.construct_tree_single_node(
                                    pt, parse_node_id, list_node_id)
                    else:
                        # assert len(l) == 1
                        # TODO: for conf_t4 just pick the first one
                        # TODO: make sure this is conf_t4!!!!
                        self.gt_nodes.append(l[0])
                        parse_node_id, list_node_id = l[0]
                        self.best_label_tree.construct_tree_single_node(
                            pt, parse_node_id, list_node_id)

            else:
                assert len(l) == 1
                pair = l[0]
                if pair is not None:
                    self.gt_nodes.append(pair)
                    parse_node_id, list_node_id = pair
                    self.best_label_tree.construct_tree_single_node(
                        pt, parse_node_id, list_node_id)
        # print("best_label_tree:", self.best_label_tree)

    def locate_gt(self, pt: ParseTree, gt_str: List[str], gt_str_multiple_option: List[List[str]]) \
            -> Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]]]:

        if gt_str[0] == '':
            return [], []
        self.duped = False
        res = [None for _ in gt_str]
        res_all_possible_sections = [[] for _ in gt_str_multiple_option]
        tmp = self.recurse(pt.start_node, pt, gt_str, res, res_all_possible_sections, gt_str_multiple_option)
        # print("res_all_possible_sections:", res_all_possible_sections)
        if self.duped:
            self.error_code = 3
            print('WARNING: duplicate ground truth found in {}' \
                  .format(pt.file_name))
        return res, res_all_possible_sections

    def recurse(self, current_node: HTMLNode, pt, gt_str: List[str],
                res: List[Tuple[int, int]], res_all_possible_sections: List[List[Tuple[int, int]]],
                gt_str_multi_option: List[str]):
        children = pt.get_children_node(current_node.id)
        node_str = ''
        child_matched = False
        if len(children) == 0:
            if current_node.is_content_list_tree:
                list_elements = pt.list_trees[current_node.content].get_children_node(
                    pt.list_trees[current_node.content].start_node.id)
                for child in list_elements:
                    next_str, next_matched = self.recurse(
                        child, pt.list_trees[current_node.content], gt_str, res, res_all_possible_sections,
                        gt_str_multi_option)
                    child_matched |= next_matched
                    if next_str:
                        if node_str.endswith(" ") or next_str.startswith(" "):
                            node_str += next_str
                        else:
                            node_str += ' ' + next_str
            else:
                if isinstance(current_node, ListNode):
                    node_str = LabelTree.clean_content(current_node.repr_content_only())
                else:
                    node_str = LabelTree.clean_content(current_node.content)
        else:
            if isinstance(current_node, ListNode):
                node_str = LabelTree.clean_content(current_node.repr_content_only())
            else:
                node_str = LabelTree.clean_content(current_node.content)
            for child in children:
                next_str, next_matched = self.recurse(child, pt, gt_str, res, res_all_possible_sections,
                                                      gt_str_multi_option)
                child_matched |= next_matched
                if next_str:
                    if node_str.endswith(" ") or next_str.startswith(" "):
                        node_str += next_str
                    else:
                        node_str += ' ' + next_str
        # Take first gt string - may not be correct when string
        # appears multiple times. This should be okay though.
        for idx, gt in enumerate(gt_str):
            gt = LabelTree.clean_content(gt)
            # print("gt:", gt)
            # print("node_Str:", node_str)
            if gt in node_str:
                if not child_matched:
                    if res[idx] is None:
                        if isinstance(pt, ListTree):
                            res[idx] = (pt.start_node.id, current_node.id)
                        else:
                            res[idx] = (current_node.id, 0)
                    else:
                        self.duped = True

        any_child_matched = False
        for idx, gts in enumerate(gt_str_multi_option):
            for gt in gts:
                gt = LabelTree.clean_content(gt)
                # print("node_str:", node_str)
                # print("gt:", gt)
                if gt in node_str:

                    if "Checking Type Safety of Foreign Function Calls" in node_str and "TOPLAS" in node_str:  # fac_27
                        continue

                    if not child_matched:
                        any_child_matched = True

                        if isinstance(pt, ListTree):
                            if (pt.start_node.id, current_node.id) not in res_all_possible_sections[idx]:
                                res_all_possible_sections[idx].append((pt.start_node.id, current_node.id))
                        else:
                            tmp1 = [n1 for n1, _ in res_all_possible_sections[idx]]
                            tmp2 = [int(int(n1) / 1000) for n1, _ in res_all_possible_sections[idx]]
                            existing_node_check = tmp1 + tmp2
                            if current_node.id not in existing_node_check:
                                # print(current_node.id, ",", existing_node_check)
                                if (current_node.id, 0) not in res_all_possible_sections[idx]:
                                    res_all_possible_sections[idx].append((current_node.id, 0))
        child_matched = any_child_matched | child_matched
        return node_str, child_matched

    @staticmethod
    def clean_content(content: str):
        """
        Remove all non-alphanumeric characters and normalize spacing.
        This is to ensure we match the ground truth as desired.
        """
        if content is None:
            return ''
        res = ''
        space = False
        for c in content:
            if c.isalnum():
                res += c
                space = False
            else:
                if not space:
                    res += ' '
                    space = True
        res = res.replace('  ', ' ')
        return res.strip()

    def mk_node(self, parse_node_id, list_node_id) -> LabelNode:
        key = (parse_node_id, list_node_id)
        key_str = str(key)
        if self.node_content_to_id.get(key_str) is None:
            new_node = LabelNode(next(self.id_counter), key)
            self.node_content_to_id[key_str] = new_node.id
            self.nodes[new_node.id] = new_node
            return new_node
        else:
            return self.nodes[self.node_content_to_id[key_str]]

    def print_tree(self, node: LabelNode, indent: int):
        ret_str = "{}\n".format(node.print_node(indent))

        for child in self.get_children_node(node.id):
            ret_str += self.print_tree(child, indent + 1)
        return ret_str

    def __repr__(self):
        if self.start_node:
            return self.print_tree(self.start_node, indent=0)
        else:
            return ''

    def get_gt_lca(self):
        node = self.start_node
        if node is None:
            return None
        while len(self.get_children_node(node.id)) == 1:
            node = self.get_children_node(node.id)[0]
        return node
