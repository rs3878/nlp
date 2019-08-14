#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
import string

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    sset = wn.synsets(lemma, pos=pos)
    for synset in sset:
        for syn in synset.lemma_names():
            if "_" not in syn and syn not in possible_synonyms and syn != lemma:
                possible_synonyms.append(syn)
            elif "_" in syn and syn not in possible_synonyms and syn != lemma:
                syn = syn.replace("_"," ")
                possible_synonyms.append(syn)
    return set(possible_synonyms)

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):

    # Note that you have to sum up the occurence counts
    # for all senses of the word if the word and the target appear together in multiple synsets.
    sset = wn.synsets(context.lemma, context.pos)
    final_count = 0
    d = {}
    for synset in sset:
        for w in synset.lemmas():
            if w.name() != context.lemma:
                d[w.name()] = d.get(w.name(), 0) + w.count()
    result = max(d, key=d.get)
    if "_" in result:
        return result.replace("_"," ")
    return result # replace for part 2


def wn_simple_lesk_predictor(context):
    context_words = set(context.left_context + context.right_context)
    sset = wn.synsets(context.lemma, context.pos)

    final_count = 0
    res = None
    overall_count = 0
    overall_synset = None

    for synset in sset:
        # first collect words in the definition of the synset itself and its lemmas
        # All examples for the synset.
        definition = tokenize(synset.definition())
        result = set(definition)-set(stopwords.words('english'))
        result = result.union(set(synset.lemma_names()))
        for ex in synset.examples():
            ex = tokenize(ex)
            r = set(ex)-set(stopwords.words('english'))
            result = result.union(r)

        # add all hypernyms
        # The definition and all examples for all hypernyms of the synset.
        hypers = synset.hypernym_paths()[0]
        for hyper in hypers:
            d = tokenize(hyper.definition())
            d = set(d)-set(stopwords.words('english'))
            result = result.union(d)
            result = result.union(set(hyper.lemma_names()))

        similar = result & context_words
        if len(similar) >= final_count:
            final_count=len(similar)
            res = synset

        synset_count = 0
        for l in synset.lemmas():
            synset_count += l.count()
        if synset_count >= overall_count and len(synset.lemma_names()) > 1:
            overall_count = synset_count
            overall_synset = synset

    count_for_overall = 0
    overall_return = None
    for s in overall_synset.lemmas():
        if s.count() >= count_for_overall and s.name() != context.lemma:
            count_for_overall = s.count()
            overall_return = s.name()

    if final_count == 0:
        return overall_return

    c = 0
    final_return = None
    for r in res.lemmas():
        if r.count() >= c and r.name() != context.lemma:
            c = r.count()
            final_return = r.name()

    if final_return is None:
        return overall_return

    if "_" in final_return:
        return final_return.replace("_"," ")

    return final_return #replace for part 3
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context):
        candidates = get_candidates(context.lemma, context.pos)
        best = -2
        res = None
        for c in candidates:
            try:
                if self.model.similarity(c,context.lemma) >= best:
                    best = self.model.similarity(c,context.lemma)
                    res = c
            except:
                pass
        return res # replace for part 4

    def predict_nearest_with_context(self, context):
        sentence = np.zeros(len(self.model.wv[context.lemma]))+ self.model.wv[context.lemma]

        left = " ".join(context.left_context)
        right = " ".join(context.right_context)

        left_words = tokenize(left)
        left_words_ = []
        for word in left_words:
            if word not in set(stopwords.words('english')):
                if word in self.model.vocab:
                    left_words_.append(word)
                else:
                    left_words_.append('UNK')

        right_words = tokenize(right)
        right_words_ =[]
        for word in right_words:
            if word not in set(stopwords.words('english')):
                if word in self.model.vocab:
                    right_words_.append(word)
                else:
                    right_words_.append('UNK')

        if len(left_words_) >= 5:
            left_words_ = left_words_[-5:]
        if len(right_words_) >= 5:
            right_words_ = right_words_[:5]

        context_words = list(left_words_)+list(right_words_)
        for i in context_words:
            sentence += self.model.wv[i]

        sentence_rep = sentence/(len(left_words_)+len(right_words_)+1)

        def cos(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        candidates = get_candidates(context.lemma, context.pos)
        best = -2
        res = None
        for c in candidates:
            if c in self.model.vocab and cos(sentence_rep, self.model.wv[c]) >= best and c not in set(stopwords.words('english')):
                best = cos(sentence_rep, self.model.wv[c])
                res = c

        return res # replace for part 5

    def competition(self, context):
        sentence = np.zeros(len(self.model.wv[context.lemma])) + self.model.wv[context.lemma]

        left = " ".join(context.left_context)
        right = " ".join(context.right_context)

        left_words = tokenize(left)
        left_words_ = []
        for word in left_words:
            if word not in set(stopwords.words('english')):
                if word in self.model.vocab:
                    left_words_.append(word)
                else:
                    left_words_.append('UNK')

        right_words = tokenize(right)
        right_words_ = []
        for word in right_words:
            if word not in set(stopwords.words('english')):
                if word in self.model.vocab:
                    right_words_.append(word)
                else:
                    right_words_.append('UNK')

        if len(left_words_) >= 5:
            left_words_ = left_words_[-5:]
        if len(right_words_) >= 5:
            right_words_ = right_words_[:5]

        context_words = list(left_words_) + list(right_words_)
        for i in context_words:
            sentence += self.model.wv[i]*1/2

        sentence_rep = sentence / (len(context_words)/2 + 1)

        def cos(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        candidates = get_candidates(context.lemma, context.pos)
        best = -2
        res = None
        for c in candidates:
            if c in self.model.vocab and cos(sentence_rep, self.model.wv[c]) >= best and c not in set(
                    stopwords.words('english')):
                best = cos(sentence_rep, self.model.wv[c])
                res = c

        return res

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.competition(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

