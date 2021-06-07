import json
from typing import Dict
from lib.tree import ParseTree, ListTree


class JsonParser:
    def __init__(self):
        pass

    def run_parser(self, file_path, file_name) -> ParseTree:
        pt = self.parse_json("{}/{}.json".format(file_path, file_name))
        pt.construct_index()
        pt.file_name = file_name
        return pt

    def parse_json(self, input_doc_fn) -> ParseTree:
        self.pt: ParseTree = ParseTree()
        json_dict = self.load_json(input_doc_fn)
        self.parse_dict_to_tree(json_dict, is_root=True)
        return self.pt

    def load_json(self, input_doc_fn):
        with open(input_doc_fn) as json_file:
            data = json.load(json_file)
            return data

    def get_dumb_list_root_id(self, _id):
        # NOTE: I am assuming we won't have very large node id
        # Additionally, the special case for _id = 0 is because the previous
        # value of the id * 1000 will lead to duplicate id nodes
        # for a list which is at the root node.

        # Although this could be changed entirely, it is left as is
        # to prevent breaking the currently cached data.
        if _id == 0:
            return 999999
        return _id * 1000

    def parse_content_to_list(self, json: Dict, lt: ListTree):
        if json.get("isList") is None:
            if json.get("children") is None:
                new_node = lt.mk_node(
                    json["id"], str(json["key"]), json["content"])
            else:
                if json["content"] == "":
                    new_node = lt.mk_node(json["id"], str(json["key"]), None, empty_content=True)
                else:
                    new_node = lt.mk_node(json["id"], str(json["content"]), None)
        else:
            assert (json["isList"])

            if len(lt.nodes) == 0:
                new_node = lt.mk_node(json["id"], "ROOT", None, is_root=True)
            else:
                if json["content"] == "":
                    new_node = lt.mk_node(
                        json["id"], str(json["key"]), None, is_list=True, empty_content=True)
                else:
                    new_node = lt.mk_node(
                        json["id"], json["content"], None, is_list=True)

        if json.get("children") is None:
            return new_node
        else:
            cnodes_id = []

            child_elements = []
            for child_elem in json["children"]:
                # if child_elem["content"] == "" \
                # and child_elem.get("isList") is None:
                # child_elements.extend(child_elem["children"])
                # else:
                child_elements.append(child_elem)

            for child_elem in child_elements:
                if child_elem.get("_comment") is not None and \
                        len(child_elem.keys()) == 1:
                    continue

                child_elem_node = self.parse_content_to_list(child_elem, lt)
                lt.set_parent(new_node.id, child_elem_node.id)
                cnodes_id.append(child_elem_node.id)
            lt.set_children(new_node.id, cnodes_id)

            return new_node

    def parse_dict_to_tree(self, json: Dict, is_root=False):
        if json.get("isList") is None:
            new_node = self.pt.mk_node(
                json["id"], json["content"], is_root=is_root)

            if json.get("children") is None:
                return new_node
            else:
                cnodes_id = []
                for child in json["children"]:

                    if child.get("_comment") is not None and \
                            len(child.keys()) == 1:
                        continue

                    child_node = self.parse_dict_to_tree(child)
                    self.pt.set_parent(new_node.id, child_node.id)
                    cnodes_id.append(child_node.id)
                self.pt.set_children(new_node.id, cnodes_id)

        else:
            assert (json["isList"])

            # check if the content is empty
            if json["content"] == "":
                content_lt = ListTree(json["id"])
                self.parse_content_to_list(json, content_lt)
                new_node = self.pt.mk_list_node(
                    json["id"], content_lt, is_root=is_root)
            else:
                # create a new node with the
                new_node = self.pt.mk_node(
                    json["id"], json["content"], is_root=is_root)

                if json.get("children") is not None:
                    # NOTE: after we finish creating the header node for list,
                    # we change the root node info so that everything is consistent
                    json["id"] = self.get_dumb_list_root_id(json["id"])
                    json["content"] = ""

                    content_lt = ListTree(json["id"])
                    self.parse_content_to_list(json, content_lt)
                    new_list_node = self.pt.mk_list_node(
                        json["id"], content_lt, is_root=is_root)
                    self.pt.set_parent(new_node.id, new_list_node.id)
                    self.pt.set_children(new_node.id, [new_list_node.id])

        return new_node
