from __future__ import division

import datetime
from typing import Optional, Union, Type, List
import json

import operator
import nltk
import string

from pytraction.base import Base, Traction, TDict, TList, In, Out, Res, Arg

from ..models.nlp_models import Doc, SENTENCEEND, STOP, PUNC


def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False

_SENTENCEEND = SENTENCEEND()
_STOP = STOP()
_PUNC = PUNC()

class SentenceTokenizer(Traction):
    i_text: In[str]
    o_out: In[TList[str]]
    
    def _run(self, on_update):
        sentences = nltk.sent_tokenize(self.i_text.data)
        for s in sentences:
            self.o_out.data.append(s)


class WordTokenizer(Traction):
    i_text: In[str]
    o_out: In[TList[str]]
    
    def _run(self, on_update):
        self.o_out.data = TList[str](nltk.word_tokenize(self.i_text.data))


class WordDictionary(Traction): 
    i_words: In[TList[str]]
    o_dict: Out[TDict[str, (int, int)]]
    o_rev_dict: Out[TDict[int, str]]

    def _run(self, on_update):
        wid = 0
        for w in self.i_words.data:
            if w not in self.o_dict.data:
                self.o_dict.data[w] = (wid, 0)
                self.o_rev_dict.data[wid] = w
                wid += 1
            else:
                _wid, count = self.o_dict.data[w]
                count += 1
                self.o_dict.data[w] = (_wid, count)


class DictionarizedText(Traction):
    i_words: In[TList[str]]
    i_dict: In[TDict[str, (int, int)]]
    o_dict_text: Out[TList[int]]
    
    def _run(self, on_update):
        wid = 0
        for w in self.i_words.data:
            self.o_dict_text.data.append(self.i_dict.data[w][0])


class StopWordReplacer(Traction):
    i_text: In[TList[int, STOP, PUNC]]
    i_rev_dict: In[TDict[int, str]]
    o_out: In[TList[Union[int, STOP, PUNC]]]
    _stopwords: List[str] = []

    @property
    def stopwords(self):
        if not self._stopwords:
            self._stopwords = set(nltk.corpus.stopwords.words())
        return self._stopwords
    
    def _run(self, on_update):
        for wi in self.i_text.data:
            if wi == _STOP or wi == _PUNC:
                continue
            w = self.i_rev_dict.data[wi]
            if w in self.stopwords:
                self.o_out.data.append(_STOP)
            else:
                self.o_out.data.append(wi)


class PuncReplacer(Traction):
    i_text: In[TList[int, STOP, PUNC]]
    i_rev_dict: In[TDict[int, str]]
    o_out: In[TList[Union[int, STOP, PUNC]]]

    def _run(self, on_update):
        for wi in self.i_text.data:
            if wi == _STOP or wi == _PUNC:
                continue
            w = self.i_rev_dict.data[wi]
            if len(w) == 1 and w in string.punctuation:
                self.o_out.data.append(_PUNC)
            else:
                self.o_out.data.append(wi)


class PhraseExtractor(Traction):
    i_text: In[TList[int, STOP, PUNC]]
    o_out: In[TList[TList[int]]]

    def _run(self, on_update):
        phrase = TList[int]([])
        for wi in self.i_text.data:
            if wi == _STOP or wi == _PUNC:
                if len(phrase) > 0:
                    self.o_out.data.append(phrase)
                phrase = TList[int]([])
            else:
                phrase.append(wi)
                

class RakeKeywordExtractor(Traction):
    top_fraction: str = 1

    i_text: In[str]
    

 

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            words = map(lambda x: "|" if x in self.stopwords else x,
                nltk.word_tokenize(sentence.lower()))
        phrase = []
        for word in words:
            if word == "|" or isPunct(word):
                if len(phrase) > 0:
                    phrase_list.append(phrase)
                    phrase = []
            else:
                phrase.append(word)
        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(filter(lambda x: not isNumeric(x), phrase)) - 1
            for word in phrase:
                word_freq.inc(word)
                word_degree.inc(word, degree) # other words
            for word in word_freq.keys():
                word_degree[word] = word_degree[word] + word_freq[word] # itself
        # word score = deg(w) / freq(w)
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores[word]
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores
    
    def extract(self, text, incl_scores=False):
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(
            phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(),
            key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]
        else:
            return map(lambda x: x[0],
                sorted_phrase_scores[0:int(n_phrases/self.top_fraction)])


class FetchPRs(Traction):
    r_ghclient: Res[GitHubClient]
    r_glclient: Res[GitLabClient]
    i_since: In[Optional[str]]
    o_authored_prs: Out[TList[Out[PR]]]
    o_for_review_prs: Out[TList[Out[PR]]]
    o_last_updated: Out[str]

    d_: str = """Fetch active pull requests from GitHub and GitLab.
    Gitlab and github are queried for PRs where user is author or requested for review.
    If i_since input is provided, only PRs which where updtaed after i_since datetime will be returned.
    Traction store results into o_authored_prs and o_for_review_prs outputs which
    combines both GitHub and GitLab PRs.
    """

    d_i_since: str = "Time used as last_updated parameter for PRs fetch. Expected format: YYYY-MM-DDTHH:MM:SS."
    d_o_authored_prs: str = "List of authored PRs."
    d_o_for_review_prs: str = "List of PRs where user is requested for review."
    d_o_last_updated: str = "Timestamp of last PR fetch. Basicaly returns current time in ISO 8601 format."

    def _run(self, on_update):
        gh_prs_authored = self.r_ghclient.r.authored_active_prs(since=self.i_since.data)
        gl_prs_authored = self.r_glclient.r.authored_active_prs(since=self.i_since.data)

        gh_prs_for_review = self.r_ghclient.r.review_requested_active_prs(since=self.i_since.data)
        gl_prs_for_review = self.r_glclient.r.review_requested_active_prs(since=self.i_since.data)

        self.o_authored_prs.data = TList[Out[PR]]([
                                    Out[PR](data = x) for x in gh_prs_authored + gl_prs_authored])
        self.o_for_review_prs.data = TList[Out[PR]]([
                                    Out[PR](data = x) for x in gh_prs_for_review + gl_prs_for_review])

        self.o_last_updated.data = datetime.datetime.now().isoformat()

