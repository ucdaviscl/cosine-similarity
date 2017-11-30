import os, sys, time, gensim, nltk

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk import pos_tag, word_tokenize

# Import wvlib
sys.path.insert(0, '/home/khgkim/Desktop/cosine_similarity/wvlib')
import wvlib

start_time = time.time()

# Specify paths
path = '/home/khgkim/Desktop/cosine_similarity'
model_path = '/home/khgkim/Desktop/cosine_similarity/model/model.txt'
token_path = '/home/khgkim/Desktop/cosine_similarity/tokens.txt'

os.chdir(path)

# Initialize cosine similarity distance
distance = 0.8

if not (os.path.isfile(model_path)):
    # Train a word2vec model
    sentences = LineSentence(token_path)
    model = Word2Vec(sentences, size=300, window=4, min_count=10)
    model.wv.save_word2vec_format('model.txt', binary=False)
    del model
else:
    # Generate sentences for each unsimplified sentence 
    wv = wvlib.load(model_path)
    f = open('/data/wiki.unsimplified', 'r')
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
                # Absolute cut-off for cosine similarity distance
                if (nearest[1] >= distance):
                  # Filter open-class words
                  word_tag = nltk.pos_tag(nltk.word_tokenize(word))
                  words_tag = nltk.pos_tag(nltk.word_tokenize(nearest[0]))
                  if word_tag[0][1] and words_tag[0][1] in {'NN','NNS','RB','RBR','RBS',
                                                            'VB','VBD','VBG','VBN','VBP',
                                                            'VBZ','JJ','JJR','JJS'}:
                    words.append(nearest[0])
            sentence.append(words)
        sentences.append(sentence)
    del words
    del sentence
    del sentences

# Print execution time
print '\n--- Execution time: %.4s minutes ---' % ((time.time()-start_time)/60)
