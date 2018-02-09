import os
import sys
import time
import gensim
import nltk
import itertools
import heapq

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk import pos_tag, word_tokenize
from itertools import permutations

start_time = time.time()

# Import wvlib
sys.path.insert(0, 'wvlib')
import wvlib

# Constants
DISTANCE_NUM = 0.9
NEIGHBOR_NUM = 3
CANDIDATE_NUM = 5

# Specify paths
path = '.'
model_path = 'test_model.txt'
token_path = 'tokens.txt'
os.chdir(path)

if not (os.path.isfile(model_path)):
    # Train a word2vec model
    sentences = LineSentence(token_path)
    model = Word2Vec(sentences, size=300, window=4, min_count=10)
    model.wv.save_word2vec_format('model.txt', binary=False)
    del model
else:
    # Generate sentences for each unsimplified sentence
    wv = wvlib.load(model_path)
    f = open('test.txt', 'r')

    for line in f:
        sent = word_tokenize(line)
        sent_len = len(sent)
        # Loop through length of sentence
        for i in range(0, len(sent)):
            # Get nearest neighbors for each word
            for j in range(0, NEIGHBOR_NUM):
                nearest = wv.nearest(sent[i])[j]
                # Absolute cut-off for cosine distance
                if (nearest[1] >= DISTANCE_NUM):
                    # Filter open-class words
                    word_tag = nltk.pos_tag(nltk.word_tokenize(sent[i]))
                    neighbor_tag = nltk.pos_tag(nltk.word_tokenize(nearest[0]))
                    if word_tag[0][1] and neighbor_tag[0][1] in {'NN','NNS','RB','RBR','RBS',
                                                                 'VB','VBD','VBG','VBN','VBP',
                                                                 'VBZ','JJ','JJR','JJS'}:
                        word = []
                        word.append(nearest[0]) # word
                        word.append(nearest[1]) # distance score
                        sent.append(word)
        heap = []
        # Generate all possible sentence candidates by combining and averaging word vectors
        # Each sentence is the average of its words
        for sent_gen in itertools.permutations(sent, sent_len):
            temp = []
            score = 0
            for word in sent_gen:
                if isinstance(word, list):
                    score += word[1]
                    temp.append(word[0])
                else:
                    score += 1
                    temp.append(word)
            average = score/sent_len
            # Ignore reordering of original words
            if (average != 1):
                heapq.heappush(heap, (average, temp))
        # Output top CANDIDATE_NUM sentences
        for k in heapq.nlargest(CANDIDATE_NUM, heap):
            candidate = ' '.join(k[1])
            print candidate, k[0]

# Print execution time
print '\n--- Execution time: %.4s minutes ---' % ((time.time()-start_time)/60)
