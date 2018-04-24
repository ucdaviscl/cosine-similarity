import os
import sys
import time
import nltk
import fstwrap as fst
from nltk import pos_tag, word_tokenize

# Import wvlib
sys.path.insert(0, 'wvlib')
import wvlib
from wvlib import Vocabulary

# Initialize execution time
start_time = time.time()

# Model path
model_path = 'test_model.txt'

# Constants
DISTANCE_NUM = 0.80
NEIGHBOR_NUM = 3
CANDIDATE_NUM = 3

# POS
open_class = {'NN','NNS','RB','RBR','RBS',
              'VB','VBD','VBG','VBN','VBP',
              'VBZ','JJ','JJR','JJS'}

def main():
    wv = wvlib.load(model_path).normalize()
    fp = open('test.txt', 'r')

    for line in fp:
        sent = word_tokenize(line)
        sent.insert(0,'<s>')
        sent.append('</s>')
        prev = '<s>'

        # The FST
        f = fst.new()

        for i in range(1, len(sent) - 1):
            if (sent[i] in wv.vocab):
                f.add_arc(prev, sent[i], prev, sent[i], -1)
                for j in range(0, NEIGHBOR_NUM):
                    nearest = wv.nearest(sent[i])[j]
                    # Absolute cut-off for cosine distance
                    if (nearest[1] >= DISTANCE_NUM):
                        # Filter open-class words
                        word_tag = nltk.pos_tag(nltk.word_tokenize(sent[i]))
                        neighbor_tag = nltk.pos_tag(nltk.word_tokenize(nearest[0]))
                        if word_tag[0][1] and neighbor_tag[0][1] in open_class:
                            f.add_arc(prev, sent[i], prev, nearest[0], -nearest[1])
            else:
                f.add_arc(prev, sent[i], prev, prev, -1)

            # Update prev
            prev = sent[i]

        # Final state
        f.add_arc(prev, '</s>', '<epsilon>', '<epsilon>', 0)

        f.set_start('<s>')
        f.set_final('</s>')
                            
        f.printf()
        sp = fst.shortest_path_list(f, CANDIDATE_NUM)
        for path in sp:
            print("%.2f\t%s" % (path[0], path[2]))

# Define the main function
if __name__ == "__main__":
  main()

# Print execution time
print '\n--- Execution time: %.4s minutes ---' % ((time.time() - start_time) / 60)
