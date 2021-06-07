import re

from bs4.element import NavigableString

from ..config import LIST_DETECTION
from ..util import get_element_text


def is_list_item(element):
    for bullet in LIST_DETECTION['LIST_BULLET_POINTS']:
        txt = get_element_text(element)
        if len(txt) >= len(bullet) + 1 and \
                txt[:len(bullet) + 1] == bullet + ' ':
            return bullet
    return False


def replace_matched_list(elements, start_idx, num_elements, parent,
                         bullet, dom):
    if num_elements == 0:
        return
    new_elements = []
    previous_sibling = None
    next_sibling = None
    parent = None
    for idx in range(start_idx, start_idx + num_elements):
        element = elements[idx]
        for text in element.find_all(text=bullet):
            text.replace_with('')
            break
        element.name = 'li'
        if idx == start_idx:
            previous_sibling = element.previous_sibling
            # next_sibling = element.next_sibling
            parent = element.parent
        new_elements.append(element)
        element.extract()
    list_element = dom.new_tag('ul')
    for element in new_elements:
        list_element.append(element)
    if previous_sibling is not None:
        previous_sibling.insert_after(list_element)
    else:
        parent.insert(0, list_element)


def detect_lists(dom):
    """
    Find lists which are given with bullet points, but not actually in
    a list element.
    """

    # Check how many list items occur in the document
    total_matched = 0
    elements = dom.body.find_all()
    for element in elements:
        if is_list_item(element):
            total_matched += 1

    if total_matched >= LIST_DETECTION['MINIMUM_LIST_ITEMS']:
        for element in elements:
            # Loop over children
            matched = 0
            current_bullet = ''
            children = list(element.find_all(recursive=False))
            for idx in range(0, len(children) + 1):
                if idx < len(children) and is_list_item(children[idx]):
                    if is_list_item(children[idx]) == current_bullet:
                        matched += 1
                    else:
                        replace_matched_list(
                            children, idx - matched, matched,
                            element, current_bullet, dom)
                        matched = 1
                        current_bullet = is_list_item(children[idx])
                else:
                    replace_matched_list(
                        children, idx - matched, matched,
                        element, current_bullet, dom)
                    matched = 0
                    current_bullet = ''
        return dom


    elements = dom.body.find_all(string=re.compile("-.*"))
    list_element = []
    for element in elements:
        if str(element).strip().startswith("-"):
            if isinstance(element, NavigableString):
                element.string = str(element).strip()
            new_element = element.wrap(dom.new_tag("li"))
            list_element.append(new_element)
    

    return dom
