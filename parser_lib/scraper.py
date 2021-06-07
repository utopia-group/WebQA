import requests
import re
from bs4 import BeautifulSoup
from unicodedata import normalize

from . import config
from .parser import create_tree, instrument_tree
from .preprocess import preprocess_html
from .turndown import convert_to_markdown


def get_content(url):
    """Get raw content of webpage from url"""
    response = requests.get(url)
    return response.text


def get_dom(content, parser='lxml'):
    """Parse the raw HTML using bs4"""
    return BeautifulSoup(content, parser)


def get_markdown(link, local=True):
    """
    Given a path, get the markdown from parsing the HTML through
    an HTML to markdown tool.

    :param link: location of raw html either a web URL or local file path
    :param local: used to tell if URL is a file:// path
    """
    if not local:
        content = get_content(link)
    else:
        with open(link, 'rb') as f:
            content = f.read()
            content = content.decode('utf-8')
    dom = get_dom(content)
    dom_text = preprocess_html(dom)
    if config.DEBUG_MODE:
        print('[DEBUG] Preprocessed DOM:')
        print(dom_text)
        print()
    markdown = convert_to_markdown(dom_text)
    return markdown


def get_website(link, local=True):
    markdown = get_markdown(link, local=local)
    if config.DEBUG_MODE:
        print('[DEBUG] Markdown:')
        print(markdown)
        print()
    tree = create_tree(markdown)
    tree = instrument_tree(tree)
    return tree
