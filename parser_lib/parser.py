import re
import json
import enchant
from enum import Enum

from . import config
from .markdown_parser import parse_markdown, BlockType

word_check = enchant.Dict("en_US")

hack_1_header_check = re.compile(r"^(([a-zA-Z]{2,5})|(Spring)|(Fall)) ((\d{2,4}(-\d{2,4})?)|((\d{1,2}[a-zA-Z])?))$")
hack_2_header_check = re.compile(r".*(Jan|Feb|March|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Mon|Tue|Wed|Thu|Fri).*")
hack_3_header_check = re.compile(r"^Homework \d{1,2}$")
hack_4_header_check = re.compile(r"^(Exam|Midterm) \d{1,2}$")
hack_5_header_check = re.compile(r"^(Sec|Section) \d{1,2}$")

class NodeType(Enum):
    """
    Enum used to represent type of nodes. Currently used just for list.
    Designed to be extensible if we wanted to keep other type information,
    but might require some changes.
    """
    LIST = 0


class Node:
    """
    Class to represent nodes in the intermediate representation.
    Stores content and children along with type information.
    """
    def __init__(self, content, children=None, type=None):
        self.content = content
        self.children = [] if children is None else children
        self.type = type

    def add_child(self, child):
        """Add child to current node"""
        self.children.append(child)

    def empty(self):
        """Check if node is empty (contains no content)"""
        return not self.content

    def coalesce(self, b):
        """Merge types when removing nodes in tree"""
        if b.type is not None:
            if self.type is not None:
                print('WARNING: Possibly incorrect behavior')
            self.type = b.type
        return self

    def retract(self):
        """
        Utility to remove empty content nodes and children
        as well as to remove children which are lists
        """
        # Retract child nodes recursively and remove lists
        new_children = []
        for key, child in enumerate(self.children):
            while type(child) is list:
                child = Node('', children=child)
            retracted = child.retract()
            if retracted is not None:
                new_children.append(retracted)
        self.children = new_children
        # Remove myself if empty
        if len(self.children) <= 1 and self.empty():
            if len(self.children) == 1:
                return self.children[0].coalesce(self)
            else:
                return None
        # Remove child if empty
        if len(self.children) == 1 and self.children[0].empty():
            self.coalesce(self.children[0])
            self.children = self.children[0].children
        return self

    current_id = 0

    def construct_object(self, key=None):
        """Construct json object without children"""
        if self.content.startswith("**") and self.content.endswith("**"):
            content_tmp = self.content[2:-2]
            if content_tmp.startswith("[") and content_tmp.endswith("]"):
                obj = {'id': Node.current_id, 'content': self.content[3:-3]}
            else:
                obj = {'id': Node.current_id, 'content': self.content[2:-2]}
        elif self.content.startswith("[") and self.content.endswith("]"):
            if "[" in self.content[1:-1] or "]" in self.content[1:-1]:
                obj = {'id': Node.current_id, 'content': self.content}
            else:
                obj = {'id': Node.current_id, 'content': self.content[1:-1]}
        elif self.content.startswith("###### "):
            obj = {'id': Node.current_id, 'content': self.content[7:]}
        else:
            obj = {'id': Node.current_id, 'content': self.content}
        # Add type information
        if self.type is NodeType.LIST:
            obj['isList'] = True
        Node.current_id += 1
        if key:
            obj['key'] = key
        return obj

    def _to_json(self, key=None):
        """Helper method to recursively generate the JSON structure."""
        obj = self.construct_object(key=key)
        # Recursively build children
        while len(self.children) == 1 and type(self.children[0]) is list:
            self.children = self.children[0]
        if self.children:
            obj['children'] = []
            for key in range(len(self.children)):
                while type(self.children[key]) is list:
                    if len(self.children[key]) > 1:
                        self.children[key] = \
                            Node('', children=self.children[key])
                    elif len(self.children[key]) == 1:
                        self.children[key] = self.children[key][0]
                    else:
                        continue
                obj['children'].append(self.children[key]._to_json(key=key+1))
        return obj

    def to_json(self, pretty=False):
        """
        Convert tree rooted at self into JSON structure.
        """
        Node.current_id = 0
        if pretty:
            return json.dumps(self._to_json(), indent=4, ensure_ascii=False)
        return json.dumps(self._to_json(), ensure_ascii=False)

    def __repr__(self):
        """Generate JSON representation"""
        return self.to_json()


def process_header(header):
    """Compute a cleaned version of the header to be used for checking."""
    res = ''
    for idx in range(len(header)):
        if idx < len(header) - 1 \
                and header[idx] == '#' and re.match(r'\d+', header[idx + 1]):
            continue
        else:
            res += header[idx]
    return res


def is_valid_header(header):


    # NOTE: JOCELYN ADDED THIS
    # NOTE: if there is only one word, check if the word is a proper word.
    # This helps to not identify toolname in the publication title as a header
    if header.strip().startswith("**") and header.strip().endswith("**"):
        header_check = header.strip()[2:-2]
    else:
        header_check = header
    ch = header_check.replace(":", "")
    chsplit = ch.split()
    if len(chsplit) == 1:
        if not word_check.check(chsplit[0]):
            return False

    """Check if header matches some text followed by a colon."""
    # NOTE: JOCELYN UPDATED THIS
    # NOTE: I am not sure whether we need to detect pure numbers as headers. I
    # am also a bit hesitant to detect Spring 2014 as a header. We need some
    # examples to determine what works the best.
    pure_number = False
    for s in header_check.strip().split(' '):
        if re.match(r'\d+', s):
            pure_number = True
    
    if re.match(r'(^(\*\*)[\w \-\'()/,]+([:]?)(\*\*)([:]?)$)|'
                    r'(^(\*\*)?[\w \-\'()/,]+(\*\*)?([:])(\*\*)?$)|'
                    r'(###### [\w]+([:])?)',
                    process_header(header)) is None:
        return False

    if not "student" in header_check.strip().lower() and not len(header_check.strip().split(' ')) <= 4:
        return False
    
    if (pure_number and len(header_check.strip().split(' ')) >= 3):
        return False
    
    if not "student" in header_check.strip().lower() and not len(header_check) <= 41:
        return False
    
    if "student" in header_check.strip().lower() and not len(header_check) <= 62:
        return False
    
    if "http" in header_check:
        return False
    
    return True

    # return re.match(r'(^(\*\*)[\w \-\'()/,]+([:]?)(\*\*)([:]?)$)|'
    #                 r'(^(\*\*)?[\w \-\'()/,]+(\*\*)?([:])(\*\*)?$)',
    #                 process_header(header)) \
    #     and len(header.strip().split(' ')) <= 4 \
    #     and not (pure_number and len(header.strip().split(' ')) >= 3) \
    #     and len(header) <= 41 and "http" not in header


def is_below(header):
    """
    Check whether the current element should be below the block passed in.
    Used to know whether elements should stay included in the header we
    are currently under.
    """
    return get_header(blocks[pos]) > get_header(header)


def get_header(block):
    """
    Used as part of constructing the heirarchy. We assign a weight to elements
    which can be used to detect headers.

    We want headers in the html to have a higher weight. However if we see
    text which matches a format that may semantically indicate a header, we
    return a lower value. Otherwise we return a large number.
    """
    # Standard header
    if block.type is BlockType.HEADER:
        return block.header
    # Text followed by :
    elif is_valid_header(block.content.get_partial_clean()):
        return 7
    # Bolded text
    elif re.fullmatch(
            r'\s*\*\*\s*[\w\'\- ]+\s*\*\*\s*', block.content.get_raw()) \
            and len(block.content.get_raw()) <= 50:
        return 4
    # Any other paragraph or element
    else:
        return 100


def recurse():
    """
    Recursively take the markdown blocks and convert into desired tree format.
    """
    global pos, blocks
    res = None
    if blocks[pos].type is BlockType.PARAGRAPH:
        if is_valid_header(blocks[pos].content.get_partial_clean()):
            res = Node('{}'.format(blocks[pos].content.get_partial_clean()))
            header = blocks[pos]
            pos += 1
            while pos < len(blocks) and is_below(header):
                r = recurse()
                if r:
                    res.add_child(r)
        elif not blocks[pos].empty():
            res = Node('{}'.format(blocks[pos].content.get_clean()))
            pos += 1
        else:
            pos += 1
    elif blocks[pos].type is BlockType.HEADER:
        res = Node('{}'.format(blocks[pos].content.get_clean()))
        header = blocks[pos]
        pos += 1
        while pos < len(blocks) and is_below(header):
            r = recurse()
            if r:
                res.add_child(r)
    elif blocks[pos].type is BlockType.THEMATIC_BREAK:
        pos += 1
    elif blocks[pos].type is BlockType.LIST_ITEM:
        res = []
        children = blocks[pos].children
        org_blocks = blocks
        org_pos = pos
        blocks = children
        pos = 0
        while pos < len(blocks):
            r = recurse()
            if r:
                res.append(r)
        blocks = org_blocks
        pos = org_pos
        pos += 1
    elif blocks[pos].type is BlockType.LIST:
        res = []
        children = blocks[pos].children
        org_blocks = blocks
        org_pos = pos
        blocks = children
        pos = 0
        while pos < len(blocks):
            r = recurse()
            if r:
                res.append(r)
        blocks = org_blocks
        pos = org_pos
        pos += 1
        res = Node('', children=res, type=NodeType.LIST)
    return res


def create_tree(markdown):
    """
    Generate intermediate structure from given markdown.
    Parses the markdown and recursively constructs a tree.

    :param markdown: markdown to parse
    :returns: root of tree as a Node
    """
    global blocks, pos
    # parse markdown
    blocks = parse_markdown(markdown)
    if config.DEBUG_MODE:
        print('[DEBUG]: Parsed markdown')
        print(blocks)

    # create root node
    title = blocks[0].content.get_clean()
    root = Node(title)

    # recursively generate children
    pos = 1
    while pos < len(blocks):
        c = recurse()
        if c:
            root.add_child(c)
    

    # clean up tree
    root = root.retract()
    return root


def instrument_tree_h(node, p_node, c_idx):
    # check current node content:
    if len(node.children) == 0:
        if ":" in node.content:
            content_string = node.content.strip()

            if content_string.endswith(":") or content_string.startswith(":"):
                return

            split_str = content_string.split(":", 1)
            # check valid header
            if re.match(hack_1_header_check, split_str[0].strip()) or \
                re.match(hack_2_header_check, split_str[0].strip()) or \
                    re.match(hack_3_header_check, split_str[0].strip()) or \
                        re.match(hack_4_header_check, split_str[0].strip()) or \
                        re.match(hack_5_header_check, split_str[0].strip()):
                return 
            # if re.match(hack_2_header_check, split_str[0].strip()) or re.match(hack_3_header_check, split_str[0].strip()):
            #     new_node = Node(split_str[0] + ':', [Node(split_str[1].strip())])
            #     p_node.children[c_idx] = new_node
            #     return
            if (not (split_str[0].rstrip()[-1].isnumeric() and
                     split_str[1].lstrip()[0].isnumeric())) and \
                (re.search(r'[\w]',
                 split_str[1].rstrip()) is not None) and \
                    is_valid_header(split_str[0] + ':'):
                new_node = Node(split_str[0] + ':', [Node(split_str[1].strip())])
                p_node.children[c_idx] = new_node
    else:
        for idx, child in enumerate(node.children):
            instrument_tree_h(child, node, idx)


# TODO: reimplement here once we have list node as leaf node
def restructure_tree(p_node):
    if len(p_node.children) == 0:
        return

    new_children = []
    last_node_changed = False
    i = 0
    while i < (len(p_node.children) - 1):
        curr_node = p_node.children[i]
        next_node = p_node.children[i + 1]

        if len(curr_node.children) == 0:
            content = curr_node.content.rstrip()
            if (content.endswith(":") and is_valid_header(content)
                    and len(next_node.children) == 0) or \
                    (content.startswith("**") and content.endswith("**")):
                if i == (len(p_node.children) - 2):
                    last_node_changed = True
                curr_node.children = [next_node]
                new_children.append(curr_node)
                i += 2
                continue
            else:
                new_children.append(curr_node)
        else:
            restructure_tree(curr_node)
            new_children.append(curr_node)
        i += 1

    if not last_node_changed:
        curr_node = p_node.children[-1]
        restructure_tree(curr_node)
        new_children.append(curr_node)

    p_node.children = new_children


def always_have_one_child(curr_node, contents):
    if len(curr_node.children) == 0:
        contents.append(curr_node.content)
        return True
    
    if len(curr_node.children) > 1:
        return False
    
    if not curr_node.content.strip() == "":
        if curr_node.content.startswith("**") and curr_node.content.endswith("**"):
            contents.append(curr_node.content[2:-2])
        else:
            contents.append(curr_node.content)
    return always_have_one_child(curr_node.children[0], contents)

def merge_node_contents(contents):
    if len(contents) == 1:
        return contents[0]
    
    if contents[0].endswith(":"):
        return "{} {}".format(contents[0], ". ".join(contents[1:]))
    else:
        return ". ".join(contents)

def clean_node_content(string):
    if string.startswith("**") and string.endswith("**"):
        new_string = string[2:-2]
    else:
        new_string = string
    return new_string.strip().lower() if not new_string.strip().endswith(":") else new_string.strip().lower()[:-1]


def is_pub_header(p_node_content):
    return re.match(re.compile(r"([*][*])?publications([*][*])?"), p_node_content) is not None or \
        p_node_content == "conferences and journals" or \
        re.match(re.compile(r".*[']s publications"), p_node_content) is not None or \
        re.match(re.compile(r"[a-zA-Z]+ publications"), p_node_content) is not None or  \
        re.match(re.compile(r"[a-zA-Z]+ publications [[].*[]]"), p_node_content) is not None or  \
        re.match(re.compile(r"peer reviewed .*"), p_node_content) is not None

def publication_parse(p_node, pp_node, in_publication):
    if len(p_node.children) == 0:
        return

    # TODO: need to modify this later
    # check when we are into the publications sections
    p_node_content = clean_node_content(p_node.content)
    parent_in_publication = False

    if is_pub_header(p_node_content):

        parent_in_publication = True
    
    new_children = []
    i = 0
    changed_next = False

    while i < (len(p_node.children) - 1):
        child = p_node.children[i]
        child_next = p_node.children[i+1]

        child_node_content = clean_node_content(child.content)
        if len(child.children) == 0:
            new_children.append(child)
            # if current node is a year node, the next node is a list node with a lot of publications as children, then unwrap the next node 
            if (parent_in_publication or in_publication) and (child_node_content.isdigit() or child_node_content == "papers before 2005") and len(child_next.children) > 2:
                new_children.extend(child_next.children)
                if child.type is NodeType.LIST:
                    p_node.type = NodeType.LIST
                changed_next = True
                i += 2
                continue
            
            i += 1
            continue

        if parent_in_publication or in_publication:

            if parent_in_publication:
                if child_node_content == "" and len(child.children) > 5 and child.type is NodeType.LIST:
                    p_node.type == NodeType.LIST
            
            # # if it is a year-type header, cancel this layer, merge its children with the parent
            # if child_node_content.isdigit() or child_node_content == "under submission":
            #     # print("detect digit header")
            #     new_children.extend(child.children)
            #     if child.type is NodeType.LIST:
            #         p_node.type = NodeType.LIST
            #     i +=1 
            #     continue
            if (child_node_content.isdigit() or child_node_content == "under submission") and child_node_content in str(child.children):
                new_children.extend(child.children)
                if child.type is NodeType.LIST:
                    p_node.type = NodeType.LIST
                i +=1 
                continue

            # if this is a list node with empty content, and it has two children who are leaves 
            # and the first children is xxx year format, then create a new node that merge its content
            if child_node_content == "" and len(child.children) == 2 and re.match(re.compile(r"[a-zA-Z ]{3,10}([ ])*[0-9]{2,4}"), clean_node_content(child.children[0].content)) and \
                len(child.children[0].children) == 0 and len(child.children[1].children) == 0:
                child = Node("{}. {}".format(child.children[0].content, child.children[1].content), [])
            
            # # if it has only one child, merge 
            children_contents = []
            if always_have_one_child(child, children_contents):
                child= Node(merge_node_contents(children_contents), [])
            
            # if this is a list with lots of publications, unwrap it 
            if child_node_content == "" and len(child.children) > 5:
                publication_parse(child, p_node, parent_in_publication)
                new_children.extend(child.children)
                if child.type is NodeType.LIST:
                    p_node.type = NodeType.LIST
                i +=1 
                continue
            
        publication_parse(child, p_node, parent_in_publication)
        new_children.append(child)
        i += 1

    if not changed_next:
        child = p_node.children[-1]
        child_node_content = child.content.strip().lower() if not child.content.strip().endswith(":") else child.content.strip().lower()[:-1]
        children_contents = []

        
        if len(child.children) == 0:
            new_children.append(child)
        elif parent_in_publication or in_publication:
            if child_node_content.isdigit() or child_node_content == "under submission":
                new_children.extend(child.children)
                if child.type is NodeType.LIST:
                    p_node.type = NodeType.LIST
            elif child_node_content == "" and len(child.children) == 2 and re.match(re.compile(r"[a-zA-Z ]{3,10}([ ])*[0-9]{2,4}"), clean_node_content(child.children[0].content)) and \
                len(child.children[0].children) == 0 and len(child.children[1].children) == 0:
                child = Node("{}. {}".format(child.children[0].content, child.children[1].content), [])
                new_children.append(child)
            elif always_have_one_child(child, children_contents):
                child= Node(merge_node_contents(children_contents), [])
                new_children.append(child)
            elif child_node_content == "" and len(child.children) > 5:
                publication_parse(child, p_node, parent_in_publication)
                new_children.extend(child.children)
                if child.type is NodeType.LIST:
                    p_node.type = NodeType.LIST
            else:
                publication_parse(child, p_node, parent_in_publication)
                new_children.append(child)
        else:
            publication_parse(child, p_node, parent_in_publication)
            new_children.append(child)
    else:
        publication_parse(child, p_node, parent_in_publication)
        new_children.append(child)
        
    p_node.children = new_children
    


def instrument_tree(root):
    for idx, child in enumerate(root.children):
        instrument_tree_h(child, root, idx)

    restructure_tree(root)
    publication_parse(root, None, False)
    publication_parse(root, None, False)

    return root
