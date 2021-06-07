import re
import truecase

"""
Postprocessing on inline markdown elements after being processed.
"""

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
process_am_check = re.compile(r".*\d[ ]*[a|A][m|M].*")
process_mw_check = re.compile(r".*[a-zA-Z]MW.*")
process_sunday_check = re.compile(r".*Sun \d.*")


def truecase_content(content):
    """Use truecase library to update case in text"""
    return truecase.get_true_case(content)


def replace_year_acro(text):
    """Utility function to expand shortened years in text"""
    text = re.sub(r"[']([0-2][0-9])", r" 20\1", text)
    text = re.sub(r"[']([3-9][0-9])", r" 19\1", text)

    if not ("am" in text.lower() or "pm" in text.lower()):
        text = re.sub(r"(^|[^\w\d])(Su)([0-2][0-9])", r"\1Summer 20\3", text)
        text = re.sub(r"(^|[^\w\d])(S|Sp)[ ]?([0-2][0-9])", r"\1Spring 20\3", text)
        text = re.sub(r"(^|[^\w\d])(W|Wi)[ ]?([0-2][0-9])", r"\1Winter 20\3", text)
        text = re.sub(r"(^|[^\w\d])(F|Fa)[ ]?([0-2][0-9])", r"\1Fall 20\3", text)
        text = re.sub(r"(^|[^\w\d])(Au)([0-2][0-9])", r"\1Autumn 20\3", text)

        text = re.sub(r"(^|[^\w\d])(Su)([3-9][0-9])", r"\1Summer 19\3", text)
        text = re.sub(r"(^|[^\w\d])(S|Sp)([3-9][0-9])", r"\1Spring 19\3", text)
        text = re.sub(r"(^|[^\w\d])(W|Wi)([3-9][0-9])", r"\1Winter 19\3", text)
        text = re.sub(r"(^|[^\w\d])(F|Fa)([3-9][0-9])", r"\1Fall 19\3", text)
        text = re.sub(r"(^|[^\w\d])(Au)([3-9][0-9])", r"\1Autumn 10\3", text)

    return text


def normalize_time(text):
    string_cleaned = text
    # string_cleaned = text.replace("/", " ")
    # string_cleaned = string_cleaned.replace(",", " ")
    string_cleaned = string_cleaned.replace(" pm", "pm")
    string_cleaned = string_cleaned.replace(" Pm", "Pm")
    string_cleaned = string_cleaned.replace(" PM", "PM")
    string_cleaned = string_cleaned.replace("Mon, Wed and Thurs at", "Monday Wednesday Thursday")
    string_cleaned = string_cleaned.replace("Mon/Wed", "Monday Wednesday")
    string_cleaned = string_cleaned.replace("Tue/Thu", "Tuesday Thursday")
    string_cleaned = string_cleaned.replace("Mon ", "Monday ")
    string_cleaned = string_cleaned.replace("Mon,", "Monday ")
    string_cleaned = string_cleaned.replace("Tue ", "Tuesday ")
    string_cleaned = string_cleaned.replace("Wed ", "Wednesday ")
    string_cleaned = string_cleaned.replace("Thu ", "Thursday ")
    string_cleaned = string_cleaned.replace("Thrus ", "Thursday ")
    string_cleaned = string_cleaned.replace("TR ", "Thursday ")
    string_cleaned = string_cleaned.replace("Fri ", "Friday ")
    # string_cleaned = string_cleaned.replace("Sun ", "Sunday ")
    string_cleaned = string_cleaned.replace("Mondays", "Monday")
    string_cleaned = string_cleaned.replace("Tuesdays", "Tuesday")
    string_cleaned = string_cleaned.replace("Wednesdays", "Wednesday")
    string_cleaned = string_cleaned.replace("Thursdays", "Thursday")
    string_cleaned = string_cleaned.replace("Fridays", "Friday")
    string_cleaned = string_cleaned.replace("MWF", "Monday Wednesday Friday")
    string_cleaned = string_cleaned.replace("TuTh", "Tuesday Thursday")
    string_cleaned = string_cleaned.replace("Monday and Wednesday", "Monday Wednesday")
    string_cleaned = string_cleaned.replace("on Tuesday", "Tuesday")
    string_cleaned = string_cleaned.replace("on Monday", "Monday")
    string_cleaned = string_cleaned.replace("Monday & Wed,", "Monday Wednesday")
    string_cleaned = string_cleaned.replace("Monday & Wednesday", "Monday Wednesday")
    string_cleaned = string_cleaned.replace("Tuesday & Thursday", "Tuesday Thursday")
    string_cleaned = string_cleaned.replace("Tuesday, Thursday", "Tuesday Thursday")
    string_cleaned = string_cleaned.replace("Monday, Wednesday, and Friday,", "Monday Wednesday Friday")
    string_cleaned = string_cleaned.replace("Wednesday from", "Wednesday")

    if re.match(process_sunday_check, string_cleaned) is not None:
        string_cleaned = string_cleaned.replace("Sun ", "Sunday ")

    if re.match(process_mw_check, string_cleaned) is None:
        string_cleaned = string_cleaned.replace("MW", "Monday Wednesday")
    
    if any(e in string_cleaned for e in weekdays):
        string_cleaned = string_cleaned.replace("--", "-")
        string_cleaned = string_cleaned.replace("- ", "-")
        string_cleaned = string_cleaned.replace(" -", "-")
        string_cleaned = string_cleaned.replace(" - ", "-")
        string_cleaned = string_cleaned.replace(" ~ ", "-")
        string_cleaned = string_cleaned.replace("~", "-")
        string_cleaned = string_cleaned.replace("-", " - ")
        string_cleaned = string_cleaned.replace(" – ", " - ")
        string_cleaned = string_cleaned.replace("–", " - ")    

    if re.match(process_am_check, string_cleaned) is not None:
        string_cleaned = string_cleaned.replace(" am", "am")
        string_cleaned = string_cleaned.replace(" Am", "Am")
        string_cleaned = string_cleaned.replace(" AM", "AM")

    # 2013-present, ...
    string_cleaned = re.sub(r'(\d{4})-present', r'\1 - present', string_cleaned)

    # 2014/15, ...
    string_cleaned = re.sub(r'(\d{4})/(\d{2})', r'\1 / \2', string_cleaned)

    # \dth December
    string_cleaned = re.sub(r'(\d{1,2})th (January|February|March|April|May|June|July|August|September|October|November|December) (.*)', r'\2 \1 \3', string_cleaned)
    
    return string_cleaned


def replace_strings(content):
    """
    Perform replacements in inline content.
    """
    replace_candidates_str = [
        ("Ph.D", "PhD"), ("Ph.D.", "PhD"),
        ("PhD.", "PhD"), ("spring", "Spring"),
        ("fall", "Fall"), ("winter", "Winter"),
        ("summer", "Summer")]

    # content = content.encode("utf-8")

    for orig, rep in replace_candidates_str:
        content = content.replace(orig, rep)

    content = replace_year_acro(content)
    content = normalize_time(content)

    # return truecase_content(content)
    return content


def postprocess_markdown(text):
    """
    Processing on the inline text after we have parsed the markdown structure.
    """
    return replace_strings(text)
