from typing import List, Dict
import itertools


class Node:
    def __init__(self, _id: int, content):
        self.id: int = _id
        self.content = content


class Tree:
    def __init__(self):
        self.start_node: Node = None
        self.nodes: Dict[int, Node] = {}
        self.to_children_edges: Dict[int, List[int]] = {}
        self.to_parent_edges: Dict[int, int] = {}

        self.id_counter = itertools.count()

    def add_node(self, new_node: Node, parent: Node = None, is_root=False):

        # print(self.__dict__)

        self.nodes[new_node.id] = new_node

        if is_root:
            self.start_node = new_node
            return new_node

        if parent is not None:
            self.to_parent_edges[new_node] = parent.id
            if self.to_children_edges.get(parent.id) is None:
                self.to_children_edges[parent.id] = []
            elif new_node.id in self.to_children_edges[parent.id]:
                return new_node

            self.to_children_edges[parent.id].append(new_node.id)
            return new_node

        return new_node

    def set_parent(self, pnode_id: int, cnode_id: int):
        self.to_parent_edges[cnode_id] = pnode_id

    def set_children(self, pnode_id: int, cnodes_id: [int]):
        self.to_children_edges[pnode_id] = cnodes_id

    def get_children_node(self, pnode_id: int):

        if self.to_children_edges.get(pnode_id) is None:
            return []

        return [self.nodes[_id] for _id in self.to_children_edges[pnode_id]]

    def get_parent(self, cnode_id: int):

        pnode_id = self.to_parent_edges.get(cnode_id)

        if pnode_id is None:
            return None
        else:
            return self.nodes[pnode_id]

    def get_parent_id(self, cnode_id: int):

        pnode_id = self.to_parent_edges.get(cnode_id)

        if pnode_id is None:
            return None
        else:
            return self.nodes[pnode_id].id

    def get_node(self, node_id):

        # print("node_id:", node_id)
        # print("self.nodes:", self.nodes)

        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError
            # return self.nodes.get(0)
        else:
            return node
