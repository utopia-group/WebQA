import re
from bs4.element import NavigableString, Comment, Doctype
from .. import config
from .list import detect_lists
from .table import remove_tables


"""
Processing steps that take place before dom are processed.

Most pre-processing is attempted to take place by manipulating the dom
since this is the simplest to implement and is agnostic of the fact that
we use markdown as an intermediary as well as allowing us to use bs4 to
easily manipulate the dom.
"""
hack_1_header_check = re.compile(r"^((Teaching Assistant)|(TAs)|(Recitation)|(Office Hours)|(Lecture 01)|(Lecture 02))(:)?$")

def remove_excess_br(dom):
    brs = dom.find_all('br')

    for br in brs:
        if br.previous_sibling is None:
            continue
        if isinstance(br.previous_sibling, NavigableString):
            continue
        if br.previous_sibling.name == "b":
            br.decompose()
            continue
        if br.parent is None:
            continue
        if br.parent.name == "li":
            br.decompose()
            continue
    
    return dom


def remove_strikethroughs(dom):
    strikes = dom.find_all('s')
    for strike in strikes:
        strike.decompose()
    return dom


def remove_comments(dom):
    """
    Remove all HTML comments to prevent these from being
    parsed by turndown.
    """
    comments = dom.findAll(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    return dom


def remove_dd_tags(dom):
    """
    Turn <dd> tags into a inline element to prevent these from being
    processed into separate paragraphs.
    """
    dd_tags = dom.findAll('dd')
    for dd_tag in dd_tags:
        dd_tag.name = 'span'
    return dom


def correct_entities(dom):
    """
    Correct invalid HTML entities without a ; since these are not
    parsed correctly by bs4.
    """
    for name in config.ENTITIES_TO_CORRECT:
        texts = dom.find_all(text=lambda text: not isinstance(text, Doctype))
        for text in texts:
            match_regex = '&{}(?!;)'.format(name)
            replace = '\u00a0'.format(name)
            new_text = re.sub(match_regex, replace, text)
            text.replace_with(new_text)
    return dom


def remove_nbsp(dom):
    for text in dom.find_all(text='\u00a0'):
        text.replace_with(' ')
    return dom


def clean_tags(dom):
    """
    Remove a set of blacklisted tags. Can later add attributes, etc.
    Also cleans up entities and comments.
    """
    # Remove comments first
    dom = remove_comments(dom)
    if config.REMOVE_STRIKETHROUGHS:
        dom = remove_strikethroughs(dom)
    dom = remove_dd_tags(dom)


    blacklist = ['link', 'xml', 'style', 'script', 'img', 'footer', 'meta',
                 'nav', 'option', 'aside']
    blacklist_id = [
        re.compile('(.*-)?(nav(bar|igation)?)(-.*)?'),
        re.compile("container-footer"),
        re.compile("footer-widgets"),
        'footer', 'nav', 'access_container', 'call-us-callout', 'mmenu'
    ]
    blacklist_class = [
        'nav', 'navbar', 'navigation', 'mobile-menu-trigger', 'btn', 'service-wrapp', 'sidebar-container', 'vertical menu', 'collapsed-menu', 'header-nav', 'main-nav', 'footer-column', 'navbar-header', 'navbar-collapse', 'c-Header', 'c-Footer', 'mobileMainNav', 'logoRow', 'footerRow2', 'slick-slider', 'pageTabs', 'widget_nav_menu', 'sidebar-nav-inner', re.compile(r'.*promo_button'), re.compile(r'featured-section-[1-9]'),
        re.compile(r"footer-.*"), 'pane-node-body', 'doctor-callout', 'menus'
    ]
    blacklist_onclick = [
        re.compile(r".*toggle.*")
    ]
    blacklist_href = [
        re.compile(r"[.][.]/index[.]html.*")
    ]
    for tag in blacklist:
        for ele in dom.find_all(tag):
            ele.decompose()
    for _id in blacklist_id:
        for ele in dom.find_all(id=_id):
            ele.decompose()
    for _class in blacklist_class:
        for ele in dom.find_all(class_=_class):
            if ele.name is None:
                ele.decompose()
                continue
            if ele is None:
                continue
            if ele.name == 'body':
                continue
            if ele.get('class') is not None and len(ele['class']) > 100:
                continue
            
            ele.decompose() 
    for _onclick in blacklist_onclick:
        for ele in dom.find_all(onclick=_onclick):
            ele.decompose()
    for _href in blacklist_href:
        for ele in dom.find_all(href=_href):
            ele.decompose()
    
    # assert False

    # get rid of all hyperlinks
    for a in dom.find_all('a'):
        a.replaceWithChildren()

    # NOTE: JOCELYN ADDED THIS
    # need to process header separately
    for ele in dom.find_all("header"):
        # if the element inside header is only a string, then switch it to h1
        if (any(content.name == 'h1' for content in ele.contents)):
            continue
        if (len(ele.contents) == 1
                and isinstance(ele.contents[0], NavigableString)):
            ele.name = "h1"
            break
        if any("sponsor" in str(string).lower() for string in ele.contents):
            continue
        if any('title' in str(string).lower() for string in ele.contents):
            continue
        if any("h1" in str(string).lower() for string in ele.contents):
            continue
        all_header = True
        for elec in ele.contents:
            if elec.name is None:
                continue
            if not elec.name.startswith("h"):
                all_header = False
                breakpoint
        if all_header:
            ele.unwrap()
        else:
            ele.decompose()

    dom = correct_entities(dom)
    dom = remove_nbsp(dom)
    dom = remove_excess_br(dom)
    return dom


def detect_headers(dom):
    """
    Detect headers based on class names / style.
    """
    matches = config.DETECT_HEADERS['MATCHES']
    for replace, *match in matches:
        try:
            elements = dom.find_all(*match)
            for ele in elements:
                if ele.get('class') is not None and ele['class'][0] == 'bd-title' and len(ele['class']) == 2:
                    continue
                if ele.name.startswith('h') and not (ele.get('class') is not None and (ele['class'][0] == 'hero-heading' or re.match(r'.*page.*title',ele['class'][0]))):
                    continue
                if ele.name == 'h5' and replace == 'h4':
                    continue
                ele.name = replace
        except:
            # traceback.print_exc()
            # print("exception")
            pass
    
    # remove all empty h*
    elements = dom.find_all(re.compile(r"h[1-6]"))
    for ele in elements:
        if ele.text.strip() == '':
            ele.decompose()
    # assert False
    return dom


def instrument_dom(dom):
    # header_class_id = [re.compile('.*title.*')]

    # for tag in header_class_id:
    #     for ele in dom.find_all(class_=tag):
    #         ele.name = "h4"

    # table related
    # if in a row, the first td ends with :, mark it as a header
    rows = dom.findChildren('tr')

    # TODO: there is probably a better of doing this
    # TODO: if does not exist some header relationship,
    # just parse the row into a single list (mitra class)
    for row in rows:
        cells = row.findChildren('th')
        context = None
        insert_tag = "th"
        if len(cells) > 0:
            context = cells[0]
            text = context.get_text().rstrip()
        else:
            cells = row.findChildren('td')
            insert_tag = "td"
            if len(cells) > 0:
                text = cells[0].get_text()
                if text is not None:
                    text = text.rstrip()
                    if cells[0].has_attr('b'):
                        context = cells[0]
                    elif cells[0].text.rstrip().endswith(":"):
                        context = cells[0]
        if context is not None:
            context.decompose()
            new_tag = dom.new_tag(insert_tag)
            new_tag_h4 = dom.new_tag("h4")
            new_tag_h4.string = text
            new_tag.insert(0, new_tag_h4)
            row.insert(0, new_tag)
    


    # process header related to bold, or marron color or special class name
    # TODO: I fix the header here to be h4 this is TBD
    if config.CURR_DOMAIN == "class":
        header_tags = [
                        ('strong', None, True, True),  # enable this when parsing class benchmarks
                        ('par', None, True, True),  
                    ]
    elif config.CURR_DOMAIN == "fac":
        header_tags = [
                        ('strong', None, True, False),
                        # ('em', None, True, True)
                        ]
    else:    
        header_tags = [
                        ('strong', None, True, False),
                        ('em', None, True, True)
                        ]
                        
    header_tags.extend([
                    ('big', None, True, True),
                    (re.compile(r'font-weight: bold'), "style", False, False),
                    (re.compile(r'font-size: 24pt'), "style", True, True),    # fac_31
                    (re.compile(r'font-size: 30px'), "style", True, True),    # clinic_16
                    (re.compile(r'maroon'), "color", True, False),   # class_31 and class_32
                    (re.compile(r'listcaption'), "class", True, True),  # fac_7
                    (re.compile(r'style4'), "class", True, True),  # fac_15
                    (re.compile(r'paragraphTitle'), "class", True, True),  # conf_29
                    (re.compile(r'4'), "font", True, True)    # fac_32
                    ])
    for tag, tag_type, spec_header, force_header in header_tags:
        if tag_type is None:
            header_candidates = dom.find_all(tag)
        elif tag_type == "style":
            header_candidates = dom.find_all(style=tag)
        elif tag_type == "color":
            header_candidates = dom.find_all(color=tag)
        elif tag_type == "class":
            header_candidates = dom.find_all(class_=tag)
            header_candidates = [item for item in header_candidates if len(item["class"]) == 1]
        elif tag_type == "font":
            header_candidates = dom.find_all(size=tag)
        for c in header_candidates:

            if isinstance(tag, re.Pattern) and tag.match('style4') is not None:
                replace_tag = dom.new_tag("h2")
                replace_tag.string = c.text.rstrip()
                c.replace_with(replace_tag)
                continue

            if c.text.rstrip() == "":
                continue
            
            if tag == "em" and ((not (c.next_sibling is None and c.previous_sibling is None)) or c.parent.name == "strong"):
                continue

            if tag == "strong" and (c.text.strip().startswith("Mon") or c.text.strip().startswith("Wed")):    # class_25
                continue
            if tag == "strong" and c.previous_sibling is not None and isinstance(c.previous_sibling, NavigableString):
                if re.match(hack_1_header_check, c.text.rstrip()):
                    pass
                elif force_header and spec_header and c.previous_sibling.string.strip() == "" and not c.parent.name == 'a':
                    pass
                else:
                    continue
            if tag == "strong": # conf_6
                if c.previous_sibling is not None and not isinstance(c.previous_sibling, NavigableString):
                    if c.previous_sibling.previous_sibling is not None and not isinstance(c.previous_sibling.previous_sibling, NavigableString):
                        if c.previous_sibling.previous_sibling.name == "strong":
                            continue
                if c.next_sibling is not None and not isinstance(c.next_sibling, NavigableString):
                    if c.next_sibling.next_sibling is not None and not isinstance(c.next_sibling.next_sibling, NavigableString):
                        if c.next_sibling.next_sibling.name == "strong":
                            continue
                if c.parent.name == "div":  # conf_25
                    continue
            if c.text.rstrip().endswith(":"):
                replace_tag = dom.new_tag("h6")
                replace_tag.string = c.text.rstrip()
                c.replace_with(replace_tag)
            elif c.next_sibling is not None and isinstance(c.next_sibling, NavigableString) and str(c.next_sibling).strip() == ":":
                replace_tag = dom.new_tag("h6")
                replace_tag.string = c.text.rstrip()
                c.replace_with(replace_tag)
            elif spec_header and (len(c.text.split(" ")) == 1 or (force_header and len(c.text.split(" ")) <= 5)):
                replace_parent = False
                cp = c.parent
                if c.parent.name == "big":
                    replace_parent = True
                    cp = c.parent.parent
                if force_header and c.name == "li": # fac_31
                    continue
                if cp is None:
                    continue
                if not force_header:
                    if not cp.name == "td":
                        continue
                    if not cp.previous_sibling is None:
                        if isinstance(cp.previous_sibling, NavigableString):
                            if not cp.previous_sibling.rstrip() == "":
                                continue
                        elif not cp.previous_sibling.text is None:
                            if not cp.previous_sibling.text.rstrip() == "":
                                continue
                # if not force_header and (not cp.name == "td" or (cp.previous_sibling is not None and not cp.previous_sibling.rstrip() == "")):
                #     continue
                replace_tag = dom.new_tag("h6")
                if not c.text.rstrip().endswith(":"):
                    replace_tag.string = c.text.rstrip() + ":"
                else:    
                    replace_tag.string = c.text.rstrip()
                if replace_parent:
                    cp.replace_with(replace_tag)
                else:
                    c.replace_with(replace_tag)                
            if c.next_sibling is not None:
                if c.next_sibling.string is not None:
                    if c.next_sibling.string.strip().startswith(":"):
                        replace_tag = dom.new_tag("h6")
                        replace_tag.string = c.text.rstrip()
                        c.replace_with(replace_tag)
    

        
    decompose_tag = [("display:none", "style"), (re.compile(r".*abstract"), "id"), (re.compile(r".*bibtex"), "id"), ("hide", "class"), ("localhref", "class")]
    for tag, tag_type in decompose_tag:
        if tag_type == "style":
            candidates = dom.find_all(style=tag)
        elif tag_type == "id":
            candidates = dom.find_all("div", id=tag)
        elif tag_type == "class":
            candidates = dom.find_all(class_=tag)
        for candidate in candidates:
            candidate.decompose()
    
    
            
    simplify_list_indent_tags = ["blockquote"]  
    for tag in simplify_list_indent_tags:
        candidates = dom.find_all(tag)
        for candidate in candidates:
            candidate.unwrap()

    simplify_navigable_string_tags = ['font']  
    for tag in simplify_navigable_string_tags:
        candidates = dom.find_all(tag)
        for candidate in candidates:
            
            if candidate.get('color') is not None and candidate['color'] == 'maroon':
                continue
            cp = candidate.parent
            candidate.replace_with(dom.new_string(candidate.text))

            if cp.name == "a":


                new_string = ""
                cp_prev_sib = cp.previous_sibling
                if isinstance(cp_prev_sib, NavigableString):
                    if isinstance(cp.next_sibling, NavigableString):
                        new_string = cp_prev_sib.string.strip() + " " + cp.text.strip() + " " + cp.next_sibling.string.strip()
                        cp.next_sibling.extract()
                    else:
                        new_string = cp_prev_sib.string.strip() + " " + cp.text.strip()
                    cp.decompose()
                    new_tag = dom.new_string(new_string)
                    cp_prev_sib.replace_with(new_tag)

    # unwrap all tag of the format <h?><p></p></h?>
    candidates = dom.find_all(re.compile('h[1-6]'))
    for candidate in candidates:
        children = list(candidate.children)
        if len(children) == 1 and  children[0].name is not None and children[0].name == 'p':
            # assert False
            children[0].unwrap()

    candidates = dom.find_all("dl")
    for candidate in candidates:
        candidate.name = "ul"
        dt_candidates = candidate.find_all("dt")
        for dt_candidate in dt_candidates:
            dt_candidate.contents[-1].insert_after(dt_candidate.findNext("dd"))
            dt_candidate.contents[-1].unwrap()
            dt_candidate.name = "li"

    simplify_to_list_class_tags = ['card'] 
    for tag in simplify_to_list_class_tags:
        candidates = dom.find_all(class_=tag)
        for candidate in candidates:
            new_tag_ul = dom.new_tag("ul")
            new_tag_li = dom.new_tag("li")
            new_tag_li.string = candidate.get_text()
            new_tag_ul.insert(0, new_tag_li)
            candidate.replace_with(new_tag_ul)

    convert_to_list_tags = ['ol']
    for tag in convert_to_list_tags:
        candidates = dom.find_all(tag)
        for candidate in candidates:
            candidate.name = 'ul'

    simplify_list_div_tags = ["ul"]  
    for tag in simplify_list_div_tags:
        candidates = dom.find_all(tag)
        for candidate in candidates:
            tag_divs = candidate.find_all("div")
            for tag_div in tag_divs:
                tag_div.unwrap()
            

    #  any h1 tags in the li automatically decreases to h2    
    downgrade_header_tags_in_list = ['ul']
    for tag in downgrade_header_tags_in_list:
        candidates = dom.find_all(tag)
        for candidate in candidates:
            sub_candidates = candidate.find_all("h1")
            for sub_candidate in sub_candidates:
                sub_candidate.name = 'h2'
            # just go ahead and unwrap any intermediate <b> in <li> 
            sub_candidates = candidate.find_all("b")
            for sub_candidate in sub_candidates:
                if sub_candidate.previous_sibling is not None and isinstance(sub_candidate.previous_sibling, NavigableString):
                    # extract br tags
                    for linebreak in sub_candidate.parent.find_all('br'):
                        linebreak.extract()
                    sub_candidate.unwrap()
    

    # replace img with alt in some the cases
    # handle conf sponsor speicifically 
    if config.CURR_DOMAIN == "conf":
        candidates = dom.find_all('img')
        for candidate in candidates:
            if candidate.get('alt') is not None and \
                ("logo" in str(candidate['src']).lower() or \
                "sponsor" in str(candidate['src']).lower() or \
                (candidate.parent.get('class')is not None and "logo" in str(candidate.parent['class']).lower()) or \
                candidate.parent.name == 'a') and \
                not (len(list(candidate.parent.next_siblings)) > 1 and list(candidate.parent.next_siblings)[1].name == "header"):
                alt_str = str(candidate['alt']).replace("logos", "")
                alt_str = alt_str.replace("Logos", "")
                alt_str = alt_str.replace("-logo", "")
                alt_str = alt_str.replace("-Logo", "")
                alt_str = alt_str.replace("logo", "")
                alt_str = alt_str.replace("Logo", "")
                new_tag = NavigableString(alt_str)
                candidate.replace_with(new_tag)
                
                continue
            


    return dom


def insert_inline_spaces(dom):
    tags = config.INLINE_SPACE_INSERTION['INLINE_ELEMENTS']
    for tag in tags:
        eles = dom.find_all(tag)
        for ele in eles:
            if ele.string:
                ele.string.replace_with(' ' + ele.string + ' ')
    return dom


def get_dom_string(dom):
    """
    Turn bs4 dom object into a string to be further processed.
    """
    content = str(dom)
    replace_strings = [(u'\u00A0', u' '), (u'\xa0', u' ')]
    for orig, repl in replace_strings:
        content = content.replace(orig, repl)
    return content


def preprocess_html(dom):
    """
    Preprocessing that manipulates the DOM before we run any parsing on it.
    """
    dom = instrument_dom(dom)
    dom = clean_tags(dom)
    if config.INLINE_SPACE_INSERTION['ENABLE']:
        dom = insert_inline_spaces(dom)    
    dom = remove_tables(dom)
    if config.LIST_DETECTION['ENABLE']:
        dom = detect_lists(dom)
    if config.DETECT_HEADERS['ENABLE']:
        dom = detect_headers(dom)
    dom_text = get_dom_string(dom)
    # assert False
    return dom_text
