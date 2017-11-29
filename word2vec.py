import os, sys, time, gensim, nltk

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
    # Generate sentences for each unsimplified sentence 
    print 'Generating sentences...'
    wv = wvlib.load(model_path)
    f = open('test.txt', 'r')
    # Create a 2D array with each row containing a sentence with nearest neighbors
    sentences = []
    for line in f:
        sentence = []
        for word in line.split():
            words = []
            words.append(word)
            # Get nearest neighbors for each word
            for i in range (0,1):
                nearest = wv.nearest(word)[i]
                words.append(nearest[0])
            sentence.append(words)
        sentences.append(sentence)         

# Print execution time
print '--- Execution time: %s minutes ---' % ((time.time() - start_time) / 60)
