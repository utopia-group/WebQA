from typing import List, Tuple, Set
from lib.utils.tree import Tree


class Example:
    def __init__(self):
        pass


class BaselineExample(Example):
    def __init__(self, name, gt, pt=None):
        super().__init__()
        self.name = name
        self.gt = gt
        self.pt = pt

    def __repr__(self):
        return '<BaselineExample {} {}...>'.format(self.name, str(self.gt)[:10])


class ExtractExample(Example):
    def __init__(self, name, gt, parse_tree):
        super().__init__()
        self.name: str = name
        self.pt = parse_tree
        self.gt: Label = gt

        self.section_locator_exec_res = None
        self.locator_prog_id = None  # for evaluator cache index

    def __repr__(self):
        return self.name


class PredExample(Example):
    def __init__(self, pos_benchmarks: List[ExtractExample], neg_benchmarks: List[ExtractExample]):
        super().__init__()
        self.pos_benchmarks = pos_benchmarks
        self.neg_benchmarks = neg_benchmarks


class Task:
    def __init__(self, q, keyword, const_str):
        self.q = q
        self.keyword = keyword
        self.const_str = const_str


class Label:
    def __init__(self, gt_str, gt_subtree: Tree, locate_gt: bool):
        self.gt_str: GroundTruth = gt_str
        self.gt_subtree: Tree = gt_subtree
        self.locate_gt: bool = locate_gt

    def contains(self, match_section_nids: Set[Tuple[int, int]], prune=False):
        # print("gt_subtree:", self.gt_subtree)
        if not self.locate_gt and not prune:
            return True
        else:
            # NOTE: test the performance after we get rid of this
            if self.gt_subtree.best_label_tree is None:
                for nid in match_section_nids:
                    if self.gt_subtree.node_content_to_id.get(str(nid)):
                        return True
            else:
                for nid in match_section_nids:
                    if self.gt_subtree.best_label_tree.node_content_to_id.get(str(nid)):
                        return True
            return False


class GroundTruth:
    class Label:
        def __init__(self, spec: str):
            self.options = spec.split('^')
            for idx, label in enumerate(self.options):
                self.options[idx] = label[1:-1]

    def __init__(self, spec: str):
        self.spec = spec
        if not spec:
            self.labels = []
        else:
            self.labels = spec.split('|')
            for idx, label in enumerate(self.labels):
                self.labels[idx] = GroundTruth.Label(label)

    def get_labels(self) -> List[str]:
        return [option for label in self.labels for option in label.options]

    def get_labels_multi_options(self) -> List[List[str]]:
        return [label.options for label in self.labels]

    def __repr__(self):
        return self.spec
