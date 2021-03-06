#!/usr/bin/env python
"""
Author: Richard Kim

Text simplification

Build lattice of original and neighboring words for each sentence.
"""

import os
import sys
import re
import time
import gensim
import nltk
import itertools
import networkx as nx
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk import pos_tag, word_tokenize
from itertools import islice
from string import digits

# Initialize execution time
start_time = time.time()

# Import wvlib
sys.path.insert(0, 'wvlib')
import wvlib
from wvlib import Vocabulary

# Constants
DISTANCE_NUM = 0.80
NEIGHBOR_NUM = 3
CANDIDATE_NUM = 3

# Specify paths
path = '.'
model_path = 'test_model.txt'
token_path = '../compling/tokenizer_tokens.txt'
os.chdir(path)

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))

def main():
    if not (os.path.isfile(model_path)):
        # Train a word2vec model
        sentences = LineSentence(token_path)
        model = Word2Vec(sentences, size=300, window=4, min_count=10)
        model.wv.save_word2vec_format('model.txt', binary=False)
        del model
    else:
        # Generate sentences for each unsimplified sentence
        wv = wvlib.load(model_path).normalize()
        f = open('test.txt', 'r')
        line_num = sum(1 for _ in f)
        f.seek(0)
        count = 0
        # Build lattice graph for every sentence
        for line in f:
            count += 1
            print "Sentence", count, "of", line_num
            sent = word_tokenize(line)
            G = nx.DiGraph()
            G.add_node("START")
            temp = []
            words = []
            # Loop through length of sentence
            for i in range(0, len(sent)):
                node = sent[i] + '*' + str(i) + '*' # Unique identifier for nodes
                G.add_node(node)
                # Connect edges
                if (i == 0):
                    G.add_edge("START", node, weight=-1)
                else:
                    prev_node = sent[i - 1] + '*' + str(i - 1) + '*'
                    G.add_edge(prev_node, node, weight=-1)
                    for w in range(0, len(words)):
                        G.add_edge(words[w], node, weight=-1)
                # "Save" nodes to connect back to the edges
                temp = words
                words = []
                # Get nearest neighbors for each word
                if (sent[i] in wv.vocab):
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
                                neighbor_node = nearest[0] + '*' + str(j) + '*' # Unique identifier for nodes
                                words.append(neighbor_node)
                                G.add_node(neighbor_node)
                                # Connect edges
                                if (i == 0):
                                    G.add_edge("START", neighbor_node, weight=-round(nearest[1], 5))
                                else:
                                    G.add_edge(prev_node, neighbor_node, weight=-round(nearest[1], 5))
                                    for t in range(0, len(temp)):
                                        G.add_edge(temp[t], neighbor_node, weight=-round(nearest[1], 5))

            # Add END node
            G.add_node("END")
            G.add_edge(node, "END", weight=-1)
            for w in range(0, len(words)):
                G.add_edge(words[w], "END", weight=-1)

            f = open('output.txt', 'a+')
            candidate_list = []

            # Write sentence candidates from the lattice to file
            # We find the shortest path because we use negative weights
            for path in k_shortest_paths(G, "START", "END", CANDIDATE_NUM):
                H = G.subgraph(path)
                # Candidate sentences only
                candidate = []
                if (len(k_shortest_paths(G, "START", "END", CANDIDATE_NUM)) == 1): # sentences with no candidates
                    for c in range(0, CANDIDATE_NUM):
                        candidate_list.append('-------')
                    continue
                if (-(len(sent) + 1) != nx.shortest_path_length(H, "START", "END", weight='weight')):
                    for word in path: # remove unique identifiers
                        candidate.append(re.sub(r'\*.*\*','',word))
                    candidate = [c.decode('utf-8') for c in candidate]
                    candidate = ' '.join(candidate[1:-1])
                    candidate_list.append(candidate)
                else:
                    candidate_list.append('-------')

                # # Draw sub-lattice
                # pos = nx.spring_layout(H)
                # new_labels = dict(map(lambda x:((x[0],x[1]), str(x[2]['weight'])), H.edges(data=True)))
                # nx.draw_networkx(H, pos=pos, font_weight='bold', font_size=15, edge_color='g')
                # nx.draw_networkx_edge_labels(H, pos=pos, font_weight='bold', width=4, edge_labels=new_labels)
                # nx.draw_networkx_edges(H, pos, with_labels=True, width=2, arrows=False)
                # plt.show()

            for sent in range(0,len(candidate_list)):
                f.write(candidate_list[sent].encode('utf-8') + '\n')
            f.close()

            # # Draw lattice
            # pos = nx.spring_layout(H)
            # new_labels = dict(map(lambda x:((x[0],x[1]), str(x[2]['weight'])), H.edges(data=True)))
            # nx.draw_networkx(H, pos=pos, font_weight='bold', font_size=15, edge_color='g')
            # nx.draw_networkx_edge_labels(H, pos=pos, font_weight='bold', width=4, edge_labels=new_labels)
            # nx.draw_networkx_edges(H, pos, with_labels=True, width=2, arrows=False)
            # plt.show()

# Define the main function
if __name__ == "__main__":
  main()

# Print execution time
print '\n--- Execution time: %.4s minutes ---' % ((time.time() - start_time) / 60)
