import os
import tempfile


def install_turndown():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'lib/turndown')
    command = 'cd \"{}\"' \
              ' && npm install turndown 1>/dev/null 2>/dev/null' \
              .format(path)
    os.system(command)


def convert_to_markdown(content):
    """
    Convert given html content to markdown using turndown library.
    """
    # Make sure turndown is installed
    install_turndown()

    inp_fname, out_fname = None, None
    try:
        # Create temporary input and output files
        inp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        inp_fname = inp.name
        inp.write(content)
        inp.close()
        out = tempfile.NamedTemporaryFile(mode='w', delete=False)
        out_fname = out.name
        out.close()

        # Run turndown
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'lib/turndown')
        command = \
            'cd \"{}\"; ./turndown.js {} {}'.format(path, inp_fname, out_fname)
        os.system(command)

        with open(out_fname, 'r') as f:
            res = f.read()
    finally:
        if inp_fname is not None:
            os.remove(inp_fname)
        if out_fname is not None:
            os.remove(out_fname)
    return res
