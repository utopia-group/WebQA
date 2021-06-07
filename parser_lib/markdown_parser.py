import re
from enum import Enum
from .postprocess import postprocess_markdown


def is_punctuation(c):
    return c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def escape_characters(content):
    ESCAPABLE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    res = []
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == '\\' and i + 1 < len(content) and content[i + 1] in ESCAPABLE:
            res.append((content[i + 1], True))
            i += 2
        else:
            res.append((ch, False))
            i += 1
    return res


def first_clean(content):
    res = []
    i = 0
    while i < len(content):
        ch = content[i]
        # elif: entity and numeric character references
        # elif: code spans
        # elif: strikethrough
        if ch == ('[', False):
            j = i
            while j < len(content) and content[j] != (']', False):
                j += 1
            if j >= len(content):
                res.append(ch)
                i += 1
            else:
                # print('found link text')
                # text = content[i + 1:j]
                k = j
                j += 1
                if j >= len(content) or content[j] != ('(', False):
                    res.append(ch)
                    i += 1
                else:
                    while j < len(content) and content[j] != (')', False):
                        j += 1
                    if j >= len(content):
                        res.append(ch)
                        i += 1
                    else:
                        # link = content[k + 2:j]
                        # print('found entire link')
                        for x in range(i + 1, k):
                            res.append(content[x])
                        i = j + 1
        else:
            res.append(ch)
            i += 1
    return res


def second_clean(content):
    res = ''
    i = 0
    while i < len(content):
        ch = content[i]
        if ch[0] in '*_' and not ch[1]:
            i += 1
        else:
            res += ch[0]
            i += 1
    res = re.sub(r'\n|\t', ' ', res)
    return res.strip()


def clean(content):
    content = second_clean(first_clean(escape_characters(content)))
    return postprocess_markdown(content)


class ListType(Enum):
    DASH = 1
    PLUS = 2
    STAR = 3
    NUMB = 4

    @staticmethod
    def get(marker):
        if marker == '*':
            return ListType.STAR
        elif marker == '+':
            return ListType.PLUS
        elif marker == '-':
            return ListType.DASH
        else:
            return ListType.NUMB


class BlockType(Enum):
    # Leaf blocks
    THEMATIC_BREAK = 1  # can interupt paragraphs
    HEADER = 2  # only atx headers can interupt paragraphs
    CODE_BLOCK = 3  # fenced code blocks can interupt paragraphs
    HTML_BLOCK = 4  # some types can interupt paragraphs
    PARAGRAPH = 5
    # Container blocks
    BLOCK_QUOTE = 6  # can interupt paragraphs
    LIST_ITEM = 7
    LIST = 8


class Inline:
    def __init__(self, content, raw=False):
        self.content = content
        self.raw = raw

    def is_header(self):
        return False

    def get_raw(self):
        return self.content
    
    def get_partial_clean(self):
        cleaned = self.get_clean()
        if self.content.strip().startswith("**") and self.content.strip().endswith("**"):
            return "**" + cleaned + "**"
        else:
            return cleaned
        # return cleaned

    def get_clean(self):
        if self.raw:
            return self.content
        else:
            lst = clean(self.content)
            res = ''
            for ele in lst:
                res += ele[0]
            return res

    def __repr__(self):
        return self.get_clean()


class Block:
    def __init__(self, content, type=BlockType.PARAGRAPH, children=None,
                 header=None, list_label=None):
        self.type = type  # type of block
        self.content = Inline(content)  # generic content of block
        self.children = children  # only for container blocks
        self.header = header  # only for header leaf blocks
        self.list_label = list_label  # only for lists

    def empty(self):
        return not self.children and not self.content.get_clean()

    def __repr__(self):
        return '<Block {} {} {} {} {}>'.format(
            self.type, self.content, self.children,
            self.header, self.list_label)


def split_lines(markdown):
    lines = re.split(r'\n', markdown)
    # lines = map(lambda l : l, lines)
    return list(lines)


def is_blank(line):
    return re.fullmatch(r'\s*', line)


def is_thematic_break():
    seen = set()
    for c in lines[pos]:
        seen.add(c)
    seen.add(' ')
    if len(seen) > 2:
        return False
    match = re.fullmatch(r' {0,3}((-|\*|_)\s*){3,}', lines[pos])
    return match


def parse_thematic_break():
    global pos
    match = is_thematic_break()
    if match:
        pos += 1
        return Block(match[1], type=BlockType.THEMATIC_BREAK)
    return None


def is_atx_header():
    match = re.fullmatch(r' {0,3}(#{1,6}) +(.*)', lines[pos])
    return match


def parse_atx_header():
    global pos
    match = is_atx_header()
    if match:
        pos += 1
        return Block(match[2], type=BlockType.HEADER, header=len(match[1]))
    return None


def is_setext_header_continuation():
    return re.fullmatch(r' {0,3}\S.*', lines[pos])


def parse_setext_header():
    global pos
    org_pos = pos
    header_content = ''
    while pos < len(lines) and is_setext_header_continuation():
        match = re.fullmatch(r' {0,3}(=|-)+ *', lines[pos])
        if match:
            pos += 1
            header_content.strip()
            return Block(header_content, type=BlockType.HEADER,
                         header=1 if match[1] == '=' else 2)
        header_content += lines[pos]
        pos += 1
    pos = org_pos
    return None


def is_paragraph_continuation():
    return pos < len(lines) and not is_blank(lines[pos]) \
        and not is_thematic_break() and not is_atx_header() \
        and not is_list_item()


def parse_paragraph():
    global pos
    content = ''
    while is_paragraph_continuation():
        content += lines[pos] + '\n'
        pos += 1
    content.strip()
    if content:
        # print("parse_paragraph:{}".format(content))
        return Block(content, type=BlockType.PARAGRAPH)
    return None


def parse_leaf_block():
    res = parse_thematic_break()
    if res:
        return res
    res = parse_atx_header()
    if res:
        return res
    res = parse_setext_header()
    if res:
        return res
    res = parse_paragraph()
    if res:
        return res
    return None


def is_list_item():
    match = re.fullmatch(
        r'( {0,3}(-|\+|\*|([0-9]{1,9}(\.|\)))) {1,4})(\S.*)',
        lines[pos])
    if not match:
        return None
    return ListType.get(match[2])  # and not is_thematic_break()


def parse_list_item():
    global pos, lines
    match = re.fullmatch(
        r'( {0,3}(-|\+|\*|([0-9]{1,9}(\.|\)))) {1,4})(\S.*)',
        lines[pos])
    if not match:
        return None
    indent = len(match[1])
    marker = match[2]
    list_lines = [match[5]]
    pos += 1
    while (pos < len(lines) and len(lines[pos]) >= indent
           and re.fullmatch(r'\s*', lines[pos][:indent])
           and not is_thematic_break()):
        list_lines.append(lines[pos][indent:])
        pos += 1
    org_pos = pos
    org_lines = lines
    pos = 0
    lines = list_lines
    children = []
    while pos < len(lines):
        r = parse_block()
        if r:
            children.append(r)
    res = Block(
        '', type=BlockType.LIST_ITEM, children=children, list_label=marker)
    pos = org_pos
    lines = org_lines
    return res


def parse_list():
    global pos
    list_type = is_list_item()
    if not list_type:
        return None
    list_items = [parse_list_item()]
    while pos < len(lines) and \
            (is_list_item() is list_type or is_blank(lines[pos])):
        while pos < len(lines) and is_blank(lines[pos]):
            pos += 1
        if pos >= len(lines) or not is_list_item():
            break
        list_items.append(parse_list_item())
    return Block('', BlockType.LIST, children=list_items)


def parse_container_block():
    res = parse_list()
    if res:
        return res


def parse_block():
    global pos
    res = parse_container_block()
    if res:
        return res
    res = parse_leaf_block()
    if res:
        return res
    pos += 1
    return None


def parse_markdown(markdown):
    global lines, pos
    lines = split_lines(markdown)
    pos = 0
    blocks = []
    while pos < len(lines):
        r = parse_block()
        if r:
            blocks.append(r)
    return blocks
