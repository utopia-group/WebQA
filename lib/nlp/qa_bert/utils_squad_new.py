# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import re
import collections
from io import open
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import torch
from transformers.tokenization_bert import BasicTokenizer
# from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from collections import OrderedDict

from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 section_mapping=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.section_mapping = section_mapping

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class SquadFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 orig_to_token_map=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.orig_to_token_map = orig_to_token_map


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


#
# RawResult = collections.namedtuple("RawResult",
#                                    ["unique_id", "start_logits", "end_logits"])

# constructs the mapping here
def read_squad_examples(input_data, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""

    input_data = input_data["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        # print("entry: ", entry)
        for paragraph in entry["paragraphs"]:

            # print("paragraph_text: ", paragraph["context"])

            paragraph_text = paragraph["context"]
            section_mapping = {}
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            section_mode = False

            i = 0
            while i < len(paragraph_text):

                if (i + 3) < len(paragraph_text) and paragraph_text[i:(i + 2)] == "[%":
                    section_match = re.search(r"\[%(\d+[-]\d+)\]", paragraph_text[i:])
                    section_string = section_match.group()
                    section_num = section_match.group(1)
                    if section_mapping.get(section_num) is None:
                        section_mapping[section_num] = [len(doc_tokens)]
                    elif len(section_mapping.get(section_num)) == 1:
                        section_mapping[section_num].append(len(doc_tokens) - 1)
                    else:
                        raise KeyError("The key of section_num is duplicated")

                    i += len(section_string)
                    continue

                c = paragraph_text[i]
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
                i += 1

            # for c in paragraph_text:
            #     if is_whitespace(c):
            #         prev_is_whitespace = True
            #     elif
            #     else:
            #         if prev_is_whitespace:
            #             doc_tokens.append(c)
            #         else:
            #             doc_tokens[-1] += c
            #         prev_is_whitespace = False
            #     char_to_word_offset.append(len(doc_tokens) - 1)

            # print(section_mapping)
            section_mapping_sorted = sorted(section_mapping.items(), key=lambda x: (x[1][1] - x[1][0]))
            # print(section_mapping_sorted)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    section_mapping=section_mapping_sorted)
                examples.append(example)
    return examples


def squad_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                       doc_stride, max_query_length,
                                       cls_token_at_end=False,
                                       cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                       sequence_a_segment_id=0, sequence_b_segment_id=1,
                                       cls_token_segment_id=0, pad_token_segment_id=0,
                                       mask_padding_with_zero=True,
                                       is_training=False, threads=1, tqdm_enabled=False):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    threads = min(threads, cpu_count())
    for (example_index, example) in enumerate(examples):

        if example_index % 100 == 0:
            logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)

        # print(query_tokens)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            # print("doc_span:", doc_span)
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            orig_to_token_map = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                mapped_orig_pos = tok_to_orig_index[split_token_index]
                token_to_orig_map[len(tokens)] = mapped_orig_pos

                if orig_to_token_map.get(mapped_orig_pos) is not None:
                    pass
                else:
                    orig_to_token_map[mapped_orig_pos] = len(tokens)

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # print("doc span: ", tokens)
            # print("section_mask:", example.section_mapping)
            # print("input_ids: ", input_ids)
            # print("recover: ", example.doc_tokens[example.section_mapping["1"][0]: example.section_mapping["1"][1]])

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info(
                    "orig_to_token_map: %s" % " ".join(["%d:%d" % (x, y) for (x, y) in orig_to_token_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            features.append(
                SquadFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    orig_to_token_map=orig_to_token_map))
            unique_id += 1

            # print("feature_i:", tokens)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if not is_training:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    return features, dataset


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def compute_predictions():
    pass


class PrelimSectionPrediction():
    def __init__(self, sec_name, start_feature_index, start_index, start_logit, end_feature_index=None, end_index=None,
                 end_logit=None):
        self.sec_name = sec_name
        self.start_feature_index = start_feature_index
        self.start_index = start_index
        self.start_logit = start_logit
        self.end_feature_index = end_feature_index
        self.end_index = end_index
        self.end_logit = end_logit

    def __repr__(self):
        return "sec_name: {}\n start_feature_idx={}     end_feature_idx={}\n start_idx={}    end_idx={}\n start_logit={}    end_logit={}".format(
            self.sec_name, self.start_feature_index, self.end_feature_index, self.start_index, self.end_index,
            self.start_logit, self.end_logit)


_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit", "start", "end", "section"])


def _get_valid_prelim_prediction(result, feature, feature_index, max_answer_length, start_indexes, end_indexes):

    prelim_predictions = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # print("here1")
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            # print("here2")

            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))

    return prelim_predictions


def _get_n_best_from_prelim_prediction(example, section_mapping, features, prelim_predictions,
                                       n_best_size, do_lower_case, verbose_logging, version_2_with_negative=False,
                                       null_start_logit=None, null_end_logit=None):
    seen_predictions = {}
    nbest = []

    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]

        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            # print("token to orig map: ", feature.token_to_orig_map)
            # print("tok_tokens: ", tok_tokens)
            # print("pred.start_indx: ", pred.start_index)
            # print("pred.end_index: ", pred.end_index)
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            # print(example.doc_tokens)
            # print("orig_doc_start: ", orig_doc_start)
            # print("orig_doc_end: ", orig_doc_end)
            # print("orig_tokens: ", orig_tokens)

            # locate section
            section = -1
            for sec_id, sec_idx in section_mapping:
                if orig_doc_start >= sec_idx[0] and orig_doc_end <= sec_idx[1]:
                    section = sec_id
                    break
            # if section == -1:
            #     raise ValueError
            # print("section: ", section)

            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            orig_doc_start = 0
            orig_doc_end = 0
            seen_predictions[final_text] = True
            section = -1

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit,
                start=orig_doc_start,
                end=orig_doc_end,
                section=section))

        # print("nbest:", nbest)
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        start=0,
                        end=0,
                        section=-1))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start=0, end=0, section=-1))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start=0, end=0, section=-1))

    return nbest


def _get_n_best_json(nbest):
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit  # soft-max prob
        output["end_logit"] = entry.end_logit
        output["start"] = entry.start
        output["end"] = entry.end
        output["section"] = entry.section
        nbest_json.append(output)

    return best_non_null_entry, nbest_json


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold, get_mapping):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    examples_sections_best_logits_mapping = {}
    if get_mapping:
        sections_n_best = 5

        for (example_index, example) in enumerate(all_examples):
            section_mapping = example.section_mapping
            # print("section mapping: ", section_mapping)

            features = example_index_to_features[example_index]
            cross_features_sec_logits = {}
            example_sections_best_logits_mapping = {}
            section_prelim_prediction = {}
            section_empty_answer_param = {}
            visited_sec = {}

            # let's first focus on the scenario where we only have one window
            for feature_idx, feature in enumerate(features):

                result = unique_id_to_result[feature.unique_id]
                feature_lowest_orig_idx = min(feature.token_to_orig_map.values())
                feature_largest_orig_idx = max(feature.token_to_orig_map.values())

                for node_id, start_end_idx in section_mapping:

                    # print("node_id:", node_id)

                    if visited_sec.get(node_id) is not None:
                        continue

                    # map start_end_idx back to token index
                    sec_tok_start = feature.orig_to_token_map.get(start_end_idx[0])
                    sec_tok_end = feature.orig_to_token_map.get(start_end_idx[1])

                    if sec_tok_end is None or sec_tok_start is None:

                        # print("feature_lowest, start_end_idx:", feature_lowest_orig_idx, start_end_idx[1])
                        # print("feature_largest, start_end_idx:", feature_largest_orig_idx, start_end_idx[0])

                        if feature_lowest_orig_idx < start_end_idx[1] and feature_largest_orig_idx < start_end_idx[0]:
                            # print("current feature doesn't contain any context for sec {}: {}".format(node_id, start_end_idx))
                            pass
                        else:
                            if cross_features_sec_logits.get(node_id) is None:
                                cross_features_sec_logits[node_id] = (start_end_idx, [])
                            cross_features_sec_logits[node_id][1].append((feature_idx, feature))
                        continue

                    # select best index within this section
                    start_indexes = _get_best_indexes_with_range(result.start_logits, 50, sec_tok_start, sec_tok_end)
                    end_indexes = _get_best_indexes_with_range(result.end_logits, 50, sec_tok_start, sec_tok_end)

                    # print("best start_indexes:", start_indexes)
                    # print("best end_indexes:", end_indexes)

                    if section_prelim_prediction.get(node_id) is None:
                        section_prelim_prediction[node_id] = []

                    section_prelim_prediction[node_id].extend(_get_valid_prelim_prediction(result, feature, feature_idx,
                                                                                           max_answer_length,
                                                                                           start_indexes,
                                                                                           end_indexes))
                    # sorted(
                    #     _get_valid_prelim_prediction(result, feature, feature_idx, max_answer_length, start_indexes,
                    #                                  end_indexes),
                    #     key=lambda x: (x.start_logit + x.end_logit),
                    #     reverse=True)

                    # if we could have irrelevant answers, get the min score of irrelevant
                    if section_empty_answer_param.get(node_id) is None:
                        section_empty_answer_param[node_id] = {'score_null': 1000000, 'min_null_feature_index': 0,
                                                               'null_start_logit': 0, 'null_end_logit': 0}
                    # score_null = 1000000  # large and positive
                    # min_null_feature_index = 0  # the paragraph slice with min null score
                    # null_start_logit = 0  # the start logit at the slice with min null score
                    # null_end_logit = 0  # the end logit at the slice with min null score
                    if version_2_with_negative:
                        feature_null_score = result.start_logits[sec_tok_start] + result.end_logits[sec_tok_start]
                        if feature_null_score < section_empty_answer_param[node_id]['score_null']:
                            section_empty_answer_param[node_id]['score_null'] = feature_null_score
                            section_empty_answer_param[node_id]['min_null_feature_index'] = feature_idx
                            section_empty_answer_param[node_id]['null_start_logit'] = result.start_logits[sec_tok_start]
                            section_empty_answer_param[node_id]['null_end_logit'] = result.end_logits[sec_tok_start]
                        # prelim_predictions.append(
                        #     _PrelimPrediction(
                        #         feature_index=min_null_feature_index,
                        #         start_index=0,
                        #         end_index=0,
                        #         start_logit=null_start_logit,
                        #         end_logit=null_end_logit))

            for node_id, prelim_predictions in section_prelim_prediction.items():
                if version_2_with_negative:
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=section_empty_answer_param[node_id]['min_null_feature_index'],
                            start_index=0,
                            end_index=0,
                            start_logit=section_empty_answer_param[node_id]['null_start_logit'],
                            end_logit=section_empty_answer_param[node_id]['null_end_logit']))
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                nbest = _get_n_best_from_prelim_prediction(example, section_mapping, features, prelim_predictions,
                                                           sections_n_best, do_lower_case, verbose_logging)
                best_non_null_entry, nbest_json = _get_n_best_json(nbest)

                example_sections_best_logits_mapping[node_id] = nbest_json
                visited_sec[node_id] = ""
                # print("appended ", node_id)

            # find best index again over those sections required multiple windows
            for node_id, values in cross_features_sec_logits.items():
                if visited_sec.get(node_id) is not None:
                    continue
                start_end_idx = values[0]
                # print("node_id, start_end_idx: {}, {}".format(node_id, start_end_idx))
                prelim_predictions = []
                # keep track of the minimum score of null start+end of position 0
                score_null = 1000000  # large and positive
                min_null_feature_index = 0  # the paragraph slice with min null score
                null_start_logit = 0  # the start logit at the slice with min null score
                null_end_logit = 0  # the end logit at the slice with min null score

                for feature_idx, feature in values[1]:
                    # print(feature_idx, feature.token_to_orig_map)
                    result = unique_id_to_result[feature.unique_id]
                    sec_tok_start = feature.orig_to_token_map.get(start_end_idx[0])
                    sec_tok_end = feature.orig_to_token_map.get(start_end_idx[1])

                    start_indexes = _get_best_indexes_with_range(result.start_logits, n_best_size,
                                                                 sec_tok_start, sec_tok_end)
                    end_indexes = _get_best_indexes_with_range(result.end_logits, n_best_size,
                                                               sec_tok_start, sec_tok_end)

                    prelim_predictions.extend(
                        _get_valid_prelim_prediction(result, feature, feature_idx, max_answer_length,
                                                     start_indexes, end_indexes))

                    if version_2_with_negative:
                        sec_tok_start_2 = 0 if sec_tok_start is None else sec_tok_start
                        feature_null_score = result.start_logits[sec_tok_start_2] + result.end_logits[sec_tok_start_2]
                        if feature_null_score < score_null:
                            score_null = feature_null_score
                            min_null_feature_index = feature_idx
                            null_start_logit = result.start_logits[sec_tok_start_2]
                            null_end_logit = result.end_logits[sec_tok_start_2]

                if version_2_with_negative:
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=min_null_feature_index,
                            start_index=0,
                            end_index=0,
                            start_logit=null_start_logit,
                            end_logit=null_end_logit))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

                nbest = _get_n_best_from_prelim_prediction(example, section_mapping, features, prelim_predictions,
                                                           sections_n_best, do_lower_case, verbose_logging)

                best_non_null_entry, nbest_json = _get_n_best_json(nbest)

                example_sections_best_logits_mapping[node_id] = nbest_json
                visited_sec[node_id] = ""

            examples_sections_best_logits_mapping[example.qas_id] = example_sections_best_logits_mapping

        # print("examples_sections_best_logits_mapping:", examples_sections_best_logits_mapping)
        # print(visited_sec)

    for (example_index, example) in enumerate(all_examples):
        section_mapping = example.section_mapping

        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            # print("result: ", result)
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            prelim_predictions.extend(_get_valid_prelim_prediction(result, feature, feature_index, max_answer_length,
                                                                   start_indexes, end_indexes))

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        nbest = _get_n_best_from_prelim_prediction(example, section_mapping, features, prelim_predictions,
                                                   n_best_size, do_lower_case, verbose_logging,
                                                   version_2_with_negative, null_start_logit, null_end_logit)
        assert len(nbest) >= 1

        best_non_null_entry, nbest_json = _get_n_best_json(nbest)
        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = (
                nbest_json[0]["text"], nbest_json[0]["start"], nbest_json[0]["end"], nbest_json[0]["section"])
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = (best_non_null_entry.text,
                                                   best_non_null_entry.start,
                                                   best_non_null_entry.end)
        all_nbest_json[example.qas_id] = nbest_json

    output_prediction_file = "output_prediction_file.json"
    output_nbest_file = "output_nbest_file.json"

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # if version_2_with_negative:
    #     with open(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return examples_sections_best_logits_mapping, all_nbest_json


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
                                           ["unique_id", "start_top_log_probs", "start_top_index",
                                            "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                               max_answer_length,
                               start_n_top, end_n_top, version_2_with_negative,
                               tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
         "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                                 end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    # with open(output_prediction_file, "w") as writer:
    #     writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # if version_2_with_negative:
    #     with open(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes_with_range(logits, n_best_size, start_idx=None, end_idx=None):
    if start_idx is None and end_idx is None:
        index_and_score = list(enumerate(logits))
    elif start_idx is None:
        index_and_score = list(enumerate(logits))[:end_idx]
    elif end_idx is None:
        index_and_score = list(enumerate(logits))[start_idx:]
    else:
        index_and_score = list(enumerate(logits))[start_idx: end_idx]
    index_and_score_sorted = sorted(index_and_score, key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score_sorted)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score_sorted[i][0])
    return best_indexes


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    # if not scores:
    # return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
