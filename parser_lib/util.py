import os


"""
Miscellaneous utilites.
"""


def get_benchmarks_by_name(benchmark_folder, match_name=None):
    """
    Find all benchmarks that match a given name in the given folder.

    :param benchmark_folder: path to folder to look for benchmarks
    :param match_name: name to look for, either string matching file
        or a string matching up to first underscore in file name. If not
        provided, all names are matched.
    """
    to_run = []

    files = [
        os.path.join(benchmark_folder, f) for f in os.listdir(benchmark_folder)
        if os.path.isfile(os.path.join(benchmark_folder, f)) and not f.startswith('.')]

    if match_name:
        for file in files:
            name = os.path.splitext(os.path.basename(file))[0]
            if '_' in match_name:
                if name == match_name:
                    to_run.append(file)
            else:
                if '_' in name:
                    if name[:name.index('_')] == match_name:
                        to_run.append(file)
                else:
                    if name == match_name:
                        to_run.append(file)
    else:
        to_run.extend(files)

    to_run.sort()
    return to_run


def is_paragraph(node):
    """
    Check if a node in dom will be parsed to a single paragraph.
    This is used to check to merge elements in a table.
    """
    tags_ignored = ['b', 'em', 'a', 'span', 'strong']
    possible_children = len(node.find_all(recursive=False))
    for tag in tags_ignored:
        possible_children -= len(node.find_all(tag, recursive=False))
    if possible_children == 0:
        return True
    return False
    # This is a bit slow, but would confirm whether this node is actually
    # going to be a paragraph.
    # s = str(node)
    # md = convert_to_markdown(s)
    # blocks = parse_markdown(md)
    # return len(blocks) == 1 and blocks[0].type == BlockType.PARAGRAPH


def get_element_text(node):
    """
    Return a cleaned version of element text which is stripped and removed
    non-blocking spaces.
    """
    res = node.text.strip()
    res = res.replace('\u00a0', ' ')
    return res
