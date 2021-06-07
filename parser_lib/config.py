import os
import re


"""
Contains important configuration values for the parsing.
"""

# Enable debugging output
DEBUG_MODE = False

# Used as default values for parsing script.
BENCHMARK_NAME = 'benchmarks'
BENCHMARK_FOLDER = 'raw'
BENCHMARK_OUTPUT = 'parsed'

# HTML entities which may be malformed in some benchmarks.
# Only included the most common and importantly nbsp.
ENTITIES_TO_CORRECT = [
    'nbsp', 'amp', 'lt', 'gt', 'quot', 'copy', 'reg', 'trade', 'hellip',
    'mdash', 'bull', 'ldquo', 'rdquo', 'lsquo', 'rsquo', 'larr',
    'rarr', 'darr', 'uarr'
]

# Parsing configuration for list detection
LIST_DETECTION = {
    # Should detect lists which have fake bullet points
    'ENABLE': True,
    # How many list elements are needed to detect a list
    'MINIMUM_LIST_ITEMS': 4,
    # Possible bullet points as a prefix of element text
    'LIST_BULLET_POINTS': [
        '*', 'o', '-', '_'
    ]
}

# Whether to add spaces between text in inline elements.
INLINE_SPACE_INSERTION = {
    'ENABLE': True,
    'MINIMUM_LENGTH': 2,
    'INLINE_ELEMENTS': [
        'strong', 'b', 'em', 'span'
    ]
}

DETECT_HEADERS = {
    'ENABLE': True,
    'MATCHES': [
        ('h1', re.compile('.*'), {'class': re.compile(r'banner-header')}),
        ('h1', re.compile('.*'), {'class': re.compile(r'fl-heading')}),
        ('h4', re.compile('.*'), {'class': re.compile(r'.*header.*')}),
        ('h5', re.compile('.*'), {'class': re.compile(r'.*subtitle.*')}),
        ('h4', re.compile('.*'), {'class': re.compile(r'.*title.*')}),
        ('h4', 'div', {'class': re.compile(r'.*title.*')}),
        ('h1', re.compile('.*'), {'class': re.compile(r'hero-heading')}),
        ('h1', re.compile('.*'), {'class': re.compile(r'.*page.*title.*')})
    ],
    'REPLACEMENT1': 'h4',

}

# Should remove <s> tags
REMOVE_STRIKETHROUGHS = True

CURR_DOMAIN = ""