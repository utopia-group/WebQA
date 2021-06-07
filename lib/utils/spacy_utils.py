import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.tokens import Token


def nlp_add_pipe(nlp, pipe):
    """
    input:
        nlp: a spacy object
        pipe: the name of the pipeline to be added
    return:
        add a pipeline into the nlp framework
    """

    if pipe == "merge_subtokens":
        nlp.add_pipe(nlp_merge_subtokens)
    else:
        nps = nlp.create_pipe(pipe)
        nlp.add_pipe(nps)


def nlp_merge_subtokens(doc, label="subtok"):
    """
    [new function for the spacy pipeline]
    input:
        doc: a document
        label: "subtok" label
    return:
        the processed document that merge all the subtokens.
    """

    merger = Matcher(doc.vocab)
    merger.add("SUBTOK", None, [{"DEP": label, "op": "+"}])
    matches = nlp_merge_common_matches(merger(doc))
    spans = [doc[start: end + 1] for _, start, end in matches]

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc


def nlp_merge_common_matches(matches):
    """
    [helper function for merge_subtokens]
    input: 
        matches: the matched spans
    return:
        find the largest common matched span, used for merge_subtokens pipe
    """

    merged_matches = []

    for idx_1, start_1, end_1 in matches:

        curr_idx = idx_1
        curr_start = start_1
        curr_end = end_1

        for idx_2, start_2, end_2 in matches:

            if (start_2 < curr_start and end_2 > curr_end) or (start_2 <= curr_start and end_2 > curr_end) or (
                    start_2 < curr_start and end_2 >= curr_end):
                curr_idx = idx_2
                curr_start = start_2
                curr_end = end_2

        merged_matches.append((curr_idx, curr_start, curr_end))

    return list(set(merged_matches))
