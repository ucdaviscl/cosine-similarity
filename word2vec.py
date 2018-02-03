import os
import sys
import time
import gensim
import nltk
import itertools

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk import pos_tag, word_tokenize
from itertools import permutations

# Import wvlib
sys.path.insert(0, '/Users/richard/Desktop/Folder/Projects/cosine_similarity/wvlib')
import wvlib

start_time = time.time()

# Specify paths
path = '/Users/richard/Desktop/Folder/Projects/cosine_similarity'
model_path = '/Users/richard/Desktop/Folder/Projects/cosine_similarity/test_model.txt'
token_path = '/Users/richard/Desktop/Folder/Projects/cosine_similarity/tokens.txt'

os.chdir(path)

# Initialize cosine similarity distance
distance = 0.9

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
        sent_tag = []
        # Loop through length of the sentence
        for x in range(0, len(sent)):
            print 'Word:', sent[x], 'Tag:', nltk.pos_tag(nltk.word_tokenize(sent[x]))[0][1]
            sent_tag.append(nltk.pos_tag(nltk.word_tokenize(sent[x]))[0][1])
            # Get nearest neighbors for each word
            for i in range(0, 3):
                nearest = wv.nearest(sent[x])[i]
                inside = False # check if only word
                # Absolute cut-off for cosine similarity distance
                if (nearest[1] >= distance):
                    # Filter open-class words
                    word_tag = nltk.pos_tag(nltk.word_tokenize(sent[x]))
                    neighbor_tag = nltk.pos_tag(nltk.word_tokenize(nearest[0]))
                    if word_tag[0][1] and neighbor_tag[0][1] in {'NN','NNS','RB','RBR','RBS',
                                                                 'VB','VBD','VBG','VBN','VBP',
                                                                 'VBZ','JJ','JJR','JJS'}:
                        sent.append(nearest[0])
                        print nearest[0], neighbor_tag[0][1]
        print sent_tag
        # Generate all possible sentences
        for sent_gen in itertools.permutations(sent, sent_len):
            count = 0
            print sent_gen
            # for i in range(0, len(sent_gen)):
            #     if nltk.pos_tag(sent_gen)[i][1] == sent_tag[i]:
            #         count = count + 1
            # if count == sent_len:
            #     print nltk.pos_tag(sent_gen)
            # count = 0

# Print execution time
print '\n--- Execution time: %.4s minutes ---' % ((time.time()-start_time)/60)
