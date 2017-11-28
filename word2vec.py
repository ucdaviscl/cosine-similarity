import os, sys, nltk, gensim

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Import wvlib
sys.path.insert(0, '/Users/richard/Desktop/Folder/Projects/cosine_similarity/wvlib')
import wvlib

# Specify paths
path = '/Users/richard/Desktop/Folder/Projects/cosine_similarity'
model_path = '/Users/richard/Desktop/Folder/Projects/cosine_similarity/model.txt'
wiki_normal = '/Users/richard/Desktop/Folder/Projects/cosine_similarity/data/wiki.unsimplified'

os.chdir(path)

if not (os.path.isfile(model_path)):
    # Train a word2vec model
    print 'Training model...'
    sentences = LineSentence(wiki_normal)
    model = Word2Vec(sentences, size=100, window=4, min_count=3)
    model.wv.save_word2vec_format('model.txt', binary=False)
else:
    # Get five nearest neighbors for each word
    print 'Generating sentences...'
    wv = wvlib.load('model.txt')
    f = open('test.txt', 'r')
    for line in f:
        for word in line.split():
            print word
            for i in range (0,5):
                print wv.nearest(word)[i]
        # print 'end of sentence'
    # del model
