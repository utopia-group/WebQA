import os
import threading

from . import config
from .scraper import get_website

"""
Utilities for running benchmarks.
Some of these are no longer in use.
"""


def generate_output(link, verbose=True, output=None, local=True):
    """
    Generate the json output for a given link and write it to a file.
    """
    if verbose:
        print('Processing', output)

    # Run benchmark
    out = get_website(link, local=local)
    if output:
        path = os.path.join('benchmarks/', output)
        with open(path, 'wb') as f:
            f.write(bytes(out.to_json(pretty=True), 'utf8'))
    else:
        print(out.to_json(pretty=True))

    if verbose:
        print('Done processing', output)
