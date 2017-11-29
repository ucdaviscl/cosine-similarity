import os, sys, time, gensim

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Import wvlib
sys.path.insert(0, '/home/khgkim/Desktop/cosine_similarity/wvlib')
import wvlib

# Specify paths
path = '/home/khgkim/Desktop/cosine_similarity'
model_path = '/home/khgkim/Desktop/cosine_similarity/model/model.txt'
token_path = '/home/khgkim/Desktop/cosine_similarity/tokens.txt'

os.chdir(path)

start_time = time.time()

if not (os.path.isfile(model_path)):
    # Train a word2vec model
    print 'Training model...'
    sentences = LineSentence(token_path)
    model = Word2Vec(sentences, size=300, window=4, min_count=10)
    model.wv.save_word2vec_format('model.txt', binary=False)
    del model
else:
    # Get five nearest neighbors for each word
    print 'Generating sentences...'
    wv = wvlib.load(model_path)
    f = open('test.txt', 'r')
    for line in f:
        for word in line.split():
            print word
            for i in range (0,3):
                print wv.nearest(word)[i]
        # end of sentence

# Print execution time
print '--- Execution time: %s minutes ---' % ((time.time() - start_time) / 60)
