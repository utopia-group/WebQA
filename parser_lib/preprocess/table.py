from bs4 import BeautifulSoup, NavigableString
import re
from ..util import is_paragraph


"""
Preprocessing steps for handling tables.
"""


def is_blank(table_row):
    for col in table_row:
        if col.text.strip():
            return False
    return True


def is_table_header(ele):
    if ele.name == 'th':
        return True
    children = list(ele.children)
    has_bold = False
    has_ital = False
    has_h = False
    text = ele.text
    for child in children:
        if not isinstance(child, NavigableString):
            if child.name == 'b' and len(child.text) > 0:
                has_bold = True

            elif child.name == 'em' and len(child.text) > 0:
                has_ital = True
            
            elif child.name.startswith('h') and len(child.text) > 0:
                has_h = True
    return (has_bold or has_ital or has_h) and len(text) <= 50 or len(text.strip()) == 0


def check_row_header(rows, force_merge=False):
    if len(rows) < 1:
        return False
    for col in rows[0]:
        if not is_table_header(col):
            return False
    # NOTE: JOCELYN: I removed this part so that now we don't detect header for tables, the reason is to handle midterm/exam task
    # I also force merge of this table if it has len(headers) >= 4 
    if any("insurance" in str(r).lower() or "plan" in str(r).lower() for r in rows[0]):
        return False
    if len(rows[0]) >= 4:
        if not force_merge:
            return False
        else:
            return True
    if not force_merge:
        return True
    if not force_merge:
        return False


def check_col_header(rows):
    for row in rows:
        if len(row) < 1:
            return False
        if not is_table_header(row[0]):
            return False
    return True


def process_row(row, dom):
    global num_rows, num_cols
    global table_rows, num_blanks
    global row_span
    row_cols = 0
    table_row = []
    for child in row.children:
        if child.name == 'td' or child.name == 'th':
            while row_cols in row_span and row_span[row_cols] >= 1:
                table_row.append(table_rows[num_rows - 1][row_cols])
                row_cols += 1
            is_header = False
            # child.name = 'div'
            colspan = 1
            if 'colspan' in child.attrs:
                colspan = int(child.attrs['colspan'])
            if 'rowspan' in child.attrs:
                row_span[row_cols] = int(child.attrs['rowspan'])
            child.attrs = {}
            table_row.append(child)
            for i in range(colspan - 1):
                table_row.append(dom.new_tag('td'))
            row_cols += 1
    num_rows += 1
    num_cols = max(num_cols, row_cols)
    for idx in row_span:
        row_span[idx] -= 1
    

    if is_blank(table_row):
        num_blanks += 1

    # Check if row is a collection of lists (handle Natish's site)
    should_split = True
    for col in table_row:
        valid_br_count = 0
        for br in col.findAll('br'):
            parents_tag_name = [t.name for t in br.parents]
            if not ("li"  in parents_tag_name or "ul"  in parents_tag_name):
                valid_br_count += 1
        if valid_br_count < 3:
            should_split = False
    if should_split:
        splits = []
        split_len = 0
        for col in table_row:
            contents = ''.join([str(x) for x in col.contents])
            contents = re.split('<br/?>', contents)
            # stderr.write(str(contents) + '\n')
            # reparse the string
            for idx, split in enumerate(contents):
                contents[idx] = BeautifulSoup(split, 'html.parser')
            splits.append(contents)
            # stderr.write(str(len(contents)) + '\n')
            split_len = max(split_len, len(contents))
        num_rows += split_len - 1
        # stderr.write(str(split_len) + '\n')
        for i in range(split_len):
            table_row = []
            for split in splits:
                if i < len(split):
                    element = dom.new_tag('td')
                    element.append(split[i])
                    # element.contents = split[i]
                    table_row.append(element)
            table_rows.append(table_row)
    else:
        table_rows.append(table_row)


def remove_tables(dom):
    """
    Remove tables since turndown does not parse these well.

    - single <tr> -> becomes list of <td> elements
    - single <td> -> becomes list of <tr> elements
    - otherwise process into nested list structure

    The above is only considering the content of the table. If there are table
    headers, these should be added as headers of each node.
    """
    global num_rows, num_cols
    global table_rows, num_blanks
    global row_span
    tables = dom.findAll('table')
    row_span = {}
    for table in tables:

        if table.name is None:
            table.decompose()
            continue

        num_rows = 0  # number of rows excluding header
        num_cols = 0  # number of cols excluding header
        row_header = []
        col_header = []
        table_rows = []
        num_blanks = 0
        
        for child1 in table.children:
            # ignore thead, tbody, and tfoot information
            if (child1.name == 'thead' or child1.name == 'tbody' or
                    child1.name == 'tfoot'):
                for child2 in child1.children:
                    if child2.name == 'tr':
                        process_row(child2, dom)
            # parse normal row
            elif child1.name == 'tr':
                process_row(child1, dom)

        # Resize rows
        new_table = []
        for row in table_rows:
            new_row = []
            for i in range(num_cols - len(row)):
                new_row.append(dom.new_tag('td'))
            new_row.extend(row)
            new_table.append(new_row)
        table_rows = new_table

        # # Remove headers
        has_row_header = check_row_header(table_rows)
        has_col_header = check_col_header(table_rows)
        force_merge = check_row_header(table_rows, force_merge=True)


        # Convert element names to div
        for row in table_rows:
            for col in row:
                col.name = 'div'


        if has_col_header:
            headers = []
            for idx, row in enumerate(table_rows):
                headers.append(row[0])
                table_rows[idx] = table_rows[idx][1:]
            num_cols -= 1
        elif has_row_header:
            headers = []
            for col in table_rows[0]:
                headers.append(col)
            table_rows = table_rows[1:]
            num_rows -= 1


        # Handle blank rows separated 'paragraphs'
        if num_blanks >= 8 and not num_blanks == 54:
            new_rows = []
            current_ele = None
            for row in table_rows:
                if is_blank(row):
                    if current_ele is not None:
                        new_rows.append(current_ele)
                    current_ele = None
                else:
                    if current_ele is None:
                        current_ele = row
                    else:
                        current_ele.extend(row)
            table_rows = new_rows

        # Check if row should be merged into csv

        for table_row in table_rows:
            should_merge = not has_row_header
            for col in table_row:
                if not is_paragraph(col):
                    should_merge = False
            if force_merge:
                should_merge = True
            if should_merge and len(table_row) > 0:
                main_ele = table_row[0]
                if main_ele.string:
                    main_ele.string.replace_with(main_ele.text.strip())
                for i in range(1, len(table_row)):
                    if table_row[i].text:
                        main_ele.append(',' + str(table_row[i].text).strip())
                    table_row[i].decompose()

        has_header = has_col_header or has_row_header

        # delete and replace entire table with nested lists
        # TODO: handle table headers
        # if num_cols == 0 and has_col_header and len(headers) == 1:
        #     # handle the cases where the whole table is header only (num_cols = 0 and nums_rows = 1)
        #     header = dom.new_tag('h4')
        #     header.append(BeautifulSoup(headers[0].text, 'html.parser'))
        #     table.replace_with(header)
        if num_rows == 0 or num_cols == 0:
            if has_header and (num_rows != 0 or num_cols != 0):
                new_table = dom.new_tag('ul')
                for header in headers:
                    header.name = 'li'
                    new_table.append(header)
                table.replaceWith(new_table)
            # table.decompose()
        elif num_rows == 1 and num_cols == 1 and not has_header:
            table.replaceWith(table_rows[0][0])
        elif num_rows == 1 and not has_header:
            new_table = dom.new_tag('ul')
            for col in table_rows[0]:
                col.name = 'li'
                new_table.append(col)
            table.replaceWith(new_table)
        elif num_cols == 1 and not has_header:
            new_table = dom.new_tag('ul')
            for row in table_rows:
                row[0].name = 'li'
                new_table.append(row[0])
            table.replaceWith(new_table)
        else:
            # Col headers assume the headers are for entire row and not
            # attributes of items in each column.
            if has_col_header:
                new_table = dom.new_tag('ul')
                for idx, row in enumerate(table_rows):
                    row_wrapper = dom.new_tag('li')
                    header = dom.new_tag('h4')
                    header.append(
                        BeautifulSoup(headers[idx].text, 'html.parser'))
                    row_wrapper.append(header)
                    new_row = dom.new_tag('ul')
                    row_wrapper.append(new_row)
                    for col in row:
                        col.name = 'li'
                        new_row.append(col)
                    new_table.append(row_wrapper)
                table.replace_with(new_table)
            elif has_row_header:
                new_table = dom.new_tag('ul')
                for row in table_rows:
                    new_row = dom.new_tag('ul')
                    row_wrapper = dom.new_tag('li')
                    row_wrapper.append(new_row)
                    for idx, col in enumerate(row):
                        col.name = 'li'
                        if idx < len(headers):
                            header = dom.new_tag('h4')
                            header.append(BeautifulSoup(
                                headers[idx].text, 'html.parser'))
                            col.insert(0, header)
                        new_row.append(col)
                    new_table.append(row_wrapper)
                table.replace_with(new_table)
            else:
                new_table = dom.new_tag('ul')
                for row in table_rows:
                    new_row = dom.new_tag('ul')
                    row_wrapper = dom.new_tag('li')
                    row_wrapper.append(new_row)
                    new_table.append(row_wrapper)
                    for col in row:
                        col.name = 'li'
                        new_row.append(col)
                table.replace_with(new_table)
    return dom
