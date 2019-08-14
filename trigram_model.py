import sys
from collections import defaultdict
import math
import random
import numpy as np
import os
import os.path

import sys
sys.path.append("hw1_data/")

"""
COMS W4705 - Natural Language Processing - Summer 2019 
Homework 1 - Trigram Language Models
Daniel Bauer
"""

# replace all the words in corpus that's not in lexicon as unknown
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile, 'r') as corpus:
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

# given corpus, find vocabs that appear more than once and call them lexicon
def get_lexicon(corpus):
    # int which essentially means that the default value of each key in word_counts will be 0
    # This is perfect for counting occurrences of items
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

# Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
# I made it python list instead of tuple
def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    # Example: n = 3, length = 4
    # we want 0:3, 1:4

    result = []
    if n == 1:
        result.append(tuple(["START"]))
        result.append(tuple(["STOP"]))

    if n == 3 and len(sequence) == 1:
        result.append(tuple(["START","START",sequence[0]]))
        result.append(tuple(["START",sequence[0],"STOP"]))
        return result

    if n > 1 :
        for i in range(n-1):
            result.append(tuple(["START"]*(n-1-i)+ sequence[:i+1]))
        result.append(tuple(sequence[-n+1:] + ["STOP"]))

    for i in range(len(sequence)-n+1):
        result.append(tuple(sequence[i:i+n]))
    return result

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        self.sentence_counts = 0
        self.word_count = 0

        for line in corpus:
            sequence = line
            self.sentence_counts +=1

            unigrams = get_ngrams(sequence, n=1)
            for gram in unigrams:
                self.word_count += 1
                self.unigramcounts[gram] +=1

            bigrams = get_ngrams(sequence, n=2)
            for gram in bigrams:
                self.bigramcounts[gram] +=1

            trigrams = get_ngrams(sequence, n=3)
            for gram in trigrams:
                self.trigramcounts[gram] +=1

        #self.unigramcounts[('START')] = self.sentence_counts *2
        self.bigramcounts[('START', 'START')] = self.sentence_counts

        #return self

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        num = self.trigramcounts[trigram]
        den = self.bigramcounts[trigram[:-1]]

        # ??? why would there be a case where trigram has length 2 ???
        if len(trigram) == 2:
            print(trigram)
            return self.raw_bigram_probability(trigram)

        # check for in consistency
        if den == 0 and num != 0:
            print(trigram, trigram[:-1], den, num)
            return 1

        # if never seen
        if den == 0 :
            return 1/len(self.bigramcounts)

        return num/den

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram == ("START", "START"):
            return 1/2
        num = self.bigramcounts[bigram]
        den = self.unigramcounts[bigram[:1]]
        if den == 0:
            return 1/len(self.unigramcounts)

        return num/den
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        num = self.unigramcounts[unigram]
        den = self.word_count
        return num/den

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = ["START", "START"]

        for i in range(t-3):
            if result[-1] == "STOP":
                break

            match = {}
            for k,v in self.trigramcounts.items():
                if k[0] == result[-2] and k[1] == result[-1]:
                    match[k[-1]] = v
            r = np.random.choice(list(match.keys()), p=np.array(list(match.values())) / np.sum(np.array(list(match.values()))))
            result.append(r)

        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3
        lambda2 = 1/3
        lambda3 = 1/3

        tri_prob = self.raw_trigram_probability(trigram)
        bi_prob = self.raw_bigram_probability(trigram[1:])
        uni_prob = self.raw_unigram_probability(trigram[2:])

        # check for incorrect probabilities
        if tri_prob > 1 or bi_prob > 1 or uni_prob >1:
            print("In correct probabilities : ", trigram, tri_prob, bi_prob, uni_prob)

        result = lambda1 * tri_prob + lambda2 * bi_prob + lambda3 * uni_prob

        return result
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        grams = get_ngrams(sentence, 3)
        p = 1

        for gram in grams:
            p *= np.longfloat(self.smoothed_trigram_probability(gram))

        return np.log2(p)


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M = 0
        prob = 0

        for line in corpus:
            M += len(line)
            M += 1 # consider "STOP"
            prob += self.sentence_logprob(line)
        result = 2**(-(prob/M))

        return result


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2 :
                correct +=1
                total +=1
            else:
                total +=1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp1 > pp2:
                correct += 1
                total += 1
            else:
                total += 1
        
        return correct/total

if __name__ == "__main__":

    #model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    model = TrigramModel("/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/brown_train.txt")
    dev_corpus = corpus_reader("/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/brown_test.txt", model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("perplexity = ", pp)
    print("perplexity of brown_train ", model.perplexity(corpus_reader("/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/brown_train.txt", model.lexicon)))
    print(model.generate_sentence())

    # Essay scoring experiment: 
    acc = essay_scoring_experiment('/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/ets_toefl_data/train_high.txt',
                                   '/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/ets_toefl_data/train_low.txt',
                                   "/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/ets_toefl_data/test_high",
                                   "/Users/roxanne/PycharmProjects/NLP/hw1/hw1_data/ets_toefl_data/test_low")
    print("accuracy", acc)

