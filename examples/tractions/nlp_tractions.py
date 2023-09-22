from __future__ import division

from typing import Optional, Union, Type, List, Tuple, TypeVar, Generic

import nltk
import string

from pytraction.base import Base, Traction, TDict, TList, In, Out, Res, Arg, STMD, STMDSingleIn, STMDExecutorType
from pytraction.tractor import Tractor

from ..models.nlp_models import Doc, Phrase, TextPhrase, ENDSENTENCE, STOP, PUNC, Word

stopwords = nltk.corpus.stopwords.words()


def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


_ENDSENTENCEEND = ENDSENTENCE()
_STOP = STOP()
_PUNC = PUNC()


class SentenceTokenizer(Traction):
    i_text: In[str]
    o_out: Out[TList[Out[str]]]

    def _run(self, on_update=None):
        sentences = nltk.sent_tokenize(self.i_text.data)
        for s in sentences:
            self.o_out.data.append(Out[str](data=s))


class WordTokenizer(Traction):
    i_text: In[str]
    o_out: Out[TList[Out[str]]]

    def _run(self, on_update=None):
        self.o_out.data = [
            Out[str](data=x.lower())
            for x in TList[str](nltk.word_tokenize(self.i_text.data))
        ]


class STMDWordTokenizer(STMD):
    _traction: Type[Traction] = WordTokenizer
    i_text: In[TList[In[str]]]
    o_out: Out[TList[Out[TList[Out[str]]]]]
    a_pool_size: Arg[int] = Arg[int](a=4)


class SentenceDictionary(Traction):
    i_words: In[TList[str]]
    o_dict: Out[TDict[str, Word]]
    o_rev_dict: Out[TDict[int, str]]

    def _run(self, on_update=None):
        wid = 0
        for w in self.i_words.data:
            if w not in self.o_dict.data:
                self.o_dict.data[w] = Word(wid=wid, count=0)
                self.o_rev_dict.data[wid] = w
                wid += 1
            else:
                self.o_dict.data[w].count += 1


class MultiSentenceDictionary(Traction):
    i_sentences: In[TList[In[TList[In[str]]]]]
    o_dict: Out[TDict[str, Word]]
    o_rev_dict: Out[TDict[int, str]]

    def _run(self, on_update=None):
        wid = 0
        for sentence in self.i_sentences.data:
            for w in sentence.data:
                if w.data not in self.o_dict.data:
                    self.o_dict.data[w.data] = Word(wid=wid, count=1)
                    self.o_rev_dict.data[wid] = w.data
                    wid += 1
                else:
                    self.o_dict.data[w.data].count += 1


class DictionarizedText(Traction):
    i_words: In[TList[In[str]]]
    i_dict: In[TDict[str, Word]]
    o_dict_text: Out[TList[int]]

    def _run(self, on_update=None):
        for w in self.i_words.data:
            self.o_dict_text.data.append(self.i_dict.data[w.data].wid)


class StopWordReplacer(Traction):
    i_text: In[TList[int, STOP, PUNC]]
    i_rev_dict: In[TDict[int, str]]
    o_out: Out[TList[Union[int, STOP, PUNC]]]
    _stopwords: TList[str] = TList[str]([])

    #@property
    #def stopwords(self):
    #    if not self._stopwords:
    #        self._stopwords = nltk.corpus.stopwords.words()
    #    return self._stopwords

    def _run(self, on_update=None):
        for wi in self.i_text.data:
            if wi in [_STOP, _PUNC]:
                continue
            w = self.i_rev_dict.data[wi]
            if w in stopwords:
                self.o_out.data.append(_STOP)
            else:
                self.o_out.data.append(wi)


class PuncReplacer(Traction):
    i_text: In[TList[int, STOP, PUNC]]
    i_rev_dict: In[TDict[int, str]]
    o_out: Out[TList[Union[int, STOP, PUNC]]]

    def _run(self, on_update = None):
        for wi in self.i_text.data:
            if wi in [_STOP, _PUNC]:
                self.o_out.data.append(wi)
                continue
            w = self.i_rev_dict.data[wi]
            if len(w) == 1 and w in string.punctuation:
                self.o_out.data.append(_PUNC)
            else:
                self.o_out.data.append(wi)


class PhraseExtractor(Traction):
    i_text: In[TList[int, STOP, PUNC]]
    o_out: Out[TList[Out[Phrase]]]

    def _run(self, on_update = None):
        phrase = TList[int]([])
        for wi in self.i_text.data:
            if wi in [_STOP, _PUNC]:
                if len(phrase) > 0:
                    self.o_out.data.append(Out[Phrase](data=Phrase(words=phrase)))
                phrase = TList[int]([])
            else:
                phrase.append(wi)


class STMDPhraseExtractor(STMD):
    _traction: Type[Traction] = PhraseExtractor
    i_text: In[TList[In[TList[int, STOP, PUNC]]]]
    o_out: Out[TList[Out[TList[Out[Phrase]]]]]
    a_pool_size: Arg[int] = Arg[int](a=4)
    a_executor_type: Arg[STMDExecutorType] = Arg[STMDExecutorType](a=STMDExecutorType.THREAD)


class PhraseScoring(Traction):
    i_phrases: In[TList[In[Phrase]]]
    i_rev_dict: In[TDict[int, str]]
    i_dict: In[TDict[str, Word]]
    o_phrases: Out[TList[Out[Phrase]]]

    def _run(self, on_update=None):
        word_degree = {}
        for phrase in self.i_phrases.data:
            degree = len(list(filter(lambda x: not isNumeric(self.i_rev_dict.data[x]), phrase.data.words))) - 1
            for word in phrase.data.words:
                word_degree.setdefault(word, 0)
                word_degree[word] += degree
        for word in self.i_dict.data.values():
            word_degree.setdefault(word.wid, 0)
            word_degree[word.wid] += word.count

        # word score = deg(w) / freq(w)
        word_scores = {}
        for word_str, word in self.i_dict.data.items():
            word_scores[word.wid] = word_degree[word.wid] / word.count

        for phrase in self.i_phrases.data:
            phrase_score = 0.0
            for wid in phrase.data.words:
                phrase_score += word_scores[wid]
            phrase.data.score = phrase_score
            self.o_phrases.data.append(phrase)


class TextProcessor(Tractor):
    i_text: In[TList[In[str]]] = In[TList[In[str]]](data=TList[In[str]]([]))
    i_dict: In[TDict[str, Word]] = In[TDict[str, Word]](data=TDict[str, Word]({}))
    i_rev_dict: In[TDict[int, str]] = In[TDict[int, str]](data=TDict[int, str]({}))

    t_dictionarized_text: DictionarizedText = DictionarizedText(uid='1', i_words=i_text, i_dict=i_dict)
    t_stop_word_replacer: StopWordReplacer = StopWordReplacer(
        uid='1',
        i_text=t_dictionarized_text.o_dict_text,
        i_rev_dict=i_rev_dict
    )
    t_punc_replacer: PuncReplacer = PuncReplacer(
        uid='1',
        i_text=t_stop_word_replacer.o_out,
        i_rev_dict=i_rev_dict
    )
    o_out: Out[TList[Union[int, STOP, PUNC]]] = t_punc_replacer.o_out


class STMDTextProcessor(STMD):
    _traction: Type[Traction] = TextProcessor
    i_text: In[TList[In[TList[In[str]]]]]
    i_dict: STMDSingleIn[TDict[str, Word]]
    i_rev_dict: STMDSingleIn[TDict[int, str]]
    o_out: Out[TList[Out[TList[Union[int, STOP, PUNC]]]]]
    a_pool_size: Arg[int] = Arg[int](a=4)
    a_executor_type: Arg[STMDExecutorType] = Arg[STMDExecutorType](a=STMDExecutorType.THREAD)


class Tokenizer(Tractor):
    i_text: In[str] = In[str](data="")

    t_sentence_tokenizer: SentenceTokenizer = SentenceTokenizer(uid='1', i_text=i_text)
    t_word_tokenizer: STMDWordTokenizer = STMDWordTokenizer(uid='1', i_text=t_sentence_tokenizer.o_out)
    t_word_dictionary: MultiSentenceDictionary = MultiSentenceDictionary(uid='1', i_sentences=t_word_tokenizer.o_out)
    t_text_processor: STMDTextProcessor = STMDTextProcessor(
        uid='1',
        i_text=t_word_tokenizer.o_out,
        i_dict=t_word_dictionary.o_dict,
        i_rev_dict=t_word_dictionary.o_rev_dict)
    o_text: Out[TList[Out[TList[Union[int, STOP, PUNC]]]]] = t_text_processor.o_out
    o_dict: Out[TDict[str, Word]] = t_word_dictionary.o_dict
    o_rev_dict: Out[TDict[int, str]] = t_word_dictionary.o_rev_dict


T = TypeVar("T")


class Flatten(Traction, Generic[T]):
    i_in: In[TList[In[TList[In[T]]]]]
    o_out: Out[TList[Out[T]]]

    def _run(self, on_update=None):
        for in_list in self.i_in.data:
            for idata in in_list.data:
                self.o_out.data.append(idata)


class TextPhrases(Traction):
    i_phrases: In[TList[In[Phrase]]]
    i_rev_dict: In[TDict[int, str]]
    o_phrases: Out[TList[Out[TextPhrase]]]

    def _run(self, on_update=None):
        for phrase in self.i_phrases.data:
            words = TList[str]([])
            for w in phrase.data.words:
                words.append(self.i_rev_dict.data[w])
            ophrase = Out[TextPhrase](data=TextPhrase(words=words, score=phrase.data.score))
            if ophrase not in self.o_phrases.data:
                self.o_phrases.data.append(ophrase)


class RakeTractor(Tractor):
    i_text: In[str] = In[str](data="")
    t_tokenizer: Tokenizer = Tokenizer(uid='1', i_text=i_text)
    t_phraser: STMDPhraseExtractor = STMDPhraseExtractor(uid='1', i_text=t_tokenizer.o_text)
    t_flat: Flatten[Phrase] = Flatten[Phrase](uid='1', i_in=t_phraser.o_out)
    t_phrase_scoring: PhraseScoring = PhraseScoring(
        uid='1',
        i_phrases=t_flat.o_out,
        i_rev_dict=t_tokenizer.o_rev_dict,
        i_dict=t_tokenizer.o_dict)
    t_text_phrases: TextPhrases = TextPhrases(
        uid='1',
        i_phrases=t_phrase_scoring.o_phrases,
        i_rev_dict=t_tokenizer.o_rev_dict
    )

    o_text_phrases: Out[TList[Out[TextPhrase]]] = t_text_phrases.o_phrases
    o_phrases: Out[TList[Out[Phrase]]] = t_phrase_scoring.o_phrases
    o_dict: Out[TDict[str, Word]] = t_tokenizer.o_dict
    o_rev_dict: Out[TDict[int, str]] = t_tokenizer.o_rev_dict


class GetTopScoredPhrases(Traction):
    a_phrase_count: Arg[int] = Arg[int](a=5)
    i_phrases: In[TList[In[TextPhrase]]]
    o_phrases: Out[TList[Out[TextPhrase]]]

    def _run(self, on_update=None):
        top_phrases = sorted(
            self.i_phrases.data,
            key=lambda x: x.data.score,
            reverse=True)[:self.a_phrase_count.a]
        self.o_phrases.data = TList[Out[TextPhrase]](top_phrases)


if __name__ == "__main__":
    rt = RakeTractor(uid='1', i_text=In[str](data="""Compatibility of systems of linear constraints over the set of natural 
numbers. Criteria of compatibility of a system of linear Diophantine 
equations, strict inequations, and nonstrict inequations are considered. 
Upper bounds for components of a minimal set of solutions and algorithms 
of construction of minimal generating sets of solutions for all types of 
systems are given. These criteria and the corresponding algorithms for 
constructing a minimal supporting set of solutions can be used in solving 
all the considered types of systems and systems of mixed types."""))
    rt.run()
    for phrase in rt.o_text_phrases.data:
        print(phrase.data.words, phrase.data.score)
