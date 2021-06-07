import itertools
import scipy
#import stanza
#from spacy_stanza import StanzaLanguage
import spacy
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, List
import lib.utils.spacy_utils as spacy_utils

from lib.nlp.qa_bert.run_bert_qa import RunBertQA


class NLPFunc:
    def __init__(self, disable=False):
        self.disable = disable
        self.sbert = None
        self.qa = None
        self.spacy = None
        # self.sbert = SBERT()
        # self.qa = QA()
        # self.spacy = SPACY()
        # print("nlp init finished")

    def get_spacy(self):
        if self.spacy is None:
            self.spacy = SPACY()
        return self.spacy

    def get_sbert(self):
        if self.sbert is None:
            self.sbert = SBERT()
        return self.sbert

    def run_sbert(self, texts, keywords):
        if self.disable:
            return
        if self.sbert is None:
            self.sbert = SBERT()
        return self.sbert.run(texts, keywords)

    # NOTE: Assumption: there will be only one question
    def run_qa_on_doc(self, contexts: [str], question: str, max_k=5) -> Dict:
        if self.disable:
            return
        if self.qa is None:
            self.qa = QA()
        results = {question: []}
        for context in contexts:
            res = self.qa.run(context, question, max_k)

            results[question].extend(res[question])

        return results

    def run_qa_on_sec_batch(self, contexts: List[str], question: str, max_k=5) -> Dict:
        if self.disable:
            return
        if self.qa is None:
            self.qa = QA()
        return self.qa.run_batch(contexts, question, max_k=max_k)

    def run_qa_on_sec(self, context: str, question: str, max_k=5) -> Dict:
        if self.disable:
            return
        if self.qa is None:
            self.qa = QA()
        return self.qa.run(context, question, max_k)

    def run_qa_on_json(self, contexts: List[str], question: str, max_k=5) -> Dict:
        if self.disable:
            return
        if self.qa is None:
            self.qa = QA()
        return self.qa.obtain_all_node_results(contexts, question, max_k)

    def spacy_init(self, text: str):
        if self.disable:
            return
        if self.spacy is None:
            self.spacy = SPACY()
        # print('spacy_init', text)
        return self.spacy.model(text)

    def run_entity(self, text, label):
        if self.disable:
            return
        raise NotImplementedError


class SBERT:
    # bert-large-nli-max-tokens
    def __init__(self, model_name="bert-large-nli-max-tokens"):
        self.model = SentenceTransformer(model_name)

    def run(self, sentences, refs):
        # TODO: need to cache
        sentence_embeddings = self.model.encode(sentences)
        refs_embedding = self.model.encode(refs)

        # scipy.spatial.distance.cdist(
        #     XA -> ma, XB -> mb, metric='euclidean', *args, **kwargs)
        distances = scipy.spatial.distance.cdist(
            refs_embedding, sentence_embeddings, "cosine")

        # convert distances to
        # {"cand_str":[("ref_str_1", score_1), ("ref_str_2", score_2))], ..}
        return_dict = {}
        for (cand_idx, cand_str), (ref_idx, ref_str) in \
                itertools.product(enumerate(sentences), enumerate(refs)):
            entry = return_dict.get(cand_str)

            score = 1 - distances[ref_idx, cand_idx] if distances[ref_idx, cand_idx] > 0.01 else 1.0

            if entry is None:
                return_dict[cand_str] = [
                    (ref_str, score)]
            else:
                entry.append((ref_str, score))

        # print(return_dict)
        return return_dict


class QA:
    def __init__(self):
        self.qa = RunBertQA()

    def obtain_all_node_results(self, contexts: List[str], question: str, max_k):

        sections_results, results = self.qa.predict_from_input_squad(contexts, question, "0", get_mapping=True)

        # assume there are only one conext
        sections_result = list(sections_results.values())[0]
        return sections_result

    def run_batch(self, contexts: List[str], question: str, max_k):
        """

        :param contexts:
        :param question:
        :param max_k:
        :return: {title: {question: prediction, dt: dt}}
        """

        res = {}
        _, results = self.qa.predict_from_input_squad(contexts, question, "0")

        for title, result in results:
            (context, qas_all) = result[0]  # only one paragraph

            curr_res = {}
            for (q, a, t, dt, _) in qas_all:
                filtered_a = []
                count = 0
                for curr_a in a:
                    if count >= max_k:
                        break
                    if curr_a["text"] == "" or curr_a['text'] == 'empty':
                        continue
                    else:
                        filtered_a.append(curr_a)
                        count += 1
                # print("run answer:", a)
                curr_res[q] = filtered_a
                curr_res['dt'] = dt
            res[title] = curr_res

        return res

    def run(self, context: str, question: str, max_k):

        # self.qa.args.n_best_size = k

        curr_res = {}
        section_results, results = self.qa.predict_from_input_squad([context], question, "0")
        _, result = \
            results[0]
        # print("result:", result)
        (text, qas_all) = result[0]

        # print(qas_all)
        for (q, a, t, dt, _) in qas_all:
            filtered_a = []
            count = 0
            for curr_a in a:
                if count >= max_k:
                    break
                if curr_a["text"] == "" or curr_a['text'] == 'empty':
                    continue
                else:
                    filtered_a.append(curr_a)
                    count += 1
            # print("run answer:", a)
            curr_res[q] = filtered_a
            curr_res['dt'] = dt

        return curr_res


class SPACY:
    def __init__(self):
        # a mix of the stanza and spacy model
        #snlp = stanza.Pipeline(lang="en", processors='tokenize, ner')
        #self.model = StanzaLanguage(snlp)
        #sentencizer = self.model.create_pipe("sentencizer")
        #self.model.add_pipe(sentencizer)
        #spacy_model = spacy.load("en_core_web_md")
        #self.model.vocab = spacy_model.vocab
        #self.model.remove_pipe("tokenizer")
        #self.model.add_pipe(spacy_model.get_pipe("tokenizer"))
       # self.model.remove_pipe("tagger")
        #self.model.add_pipe(spacy_model.get_pipe("tagger"))
        #self.model.add_pipe(spacy_model.get_pipe("parser"))
        #self.model.add_pipe(spacy_model.get_pipe("sentencizer"))
        self.model = spacy.load("en_core_web_md")
        spacy_utils.nlp_add_pipe(self.model, "merge_subtokens")

    def entity(self, text, label):
        raise NotImplementedError
