#!/usr/bin/env python
"""
Author: Richard Kim

Text simplification

Build lattice of original and neighboring words for each sentence.
"""

import os
import sys
import time
import gensim
import nltk
import networkx as nx
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk import pos_tag, word_tokenize

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
    # Build lattice graph for every sentence
    for line in f:
        sent = word_tokenize(line)
        G = nx.DiGraph()
        G.add_node("START")
        temp = []
        words = []
        # Loop through length of sentence
        for i in range(0, len(sent)):
            node = sent[i] + str(i) # Unique identifier for nodes
            G.add_node(node)
            # Connect edges
            if (i == 0):
                G.add_edge("START", node, weight=1)
            else:
                prev_node = sent[i - 1] + str(i - 1)
                G.add_edge(prev_node, node, weight=1)
                for w in range(0, len(words)):
                    G.add_edge(words[w], node, weight=1)
            # "Save" nodes to connect back to the edges
            temp = words
            words = []
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
                        neighbor_node = nearest[0] + str(j) # Unique identifier for nodes
                        words.append(neighbor_node)
                        G.add_node(neighbor_node)
                        # Connect edges
                        if (i == 0):
                            G.add_edge("START", neighbor_node, weight=round(nearest[1], 5))
                        else:
                            G.add_edge(prev_node, neighbor_node, weight=round(nearest[1], 5))
                            for t in range(0, len(temp)):
                                G.add_edge(temp[t], neighbor_node, weight=round(nearest[1], 5))
        # Add END node
        G.add_node("END")
        G.add_edge(node, "END", weight=1)
        for w in range(0, len(words)):
            G.add_edge(words[w], "END", weight=1)

        # Draw graph
        pos = nx.spring_layout(G)
        new_labels = dict(map(lambda x:((x[0],x[1]), str(x[2]['weight'])), G.edges(data=True)))
        nx.draw_networkx(G, pos=pos, font_weight='bold', font_size=15, edge_color='g')
        nx.draw_networkx_edge_labels(G, pos=pos, font_weight='bold', width=4, edge_labels=new_labels)
        nx.draw_networkx_edges(G, pos, with_labels=True, width=2, arrows=False)
        plt.show()

# Print execution time
print '\n--- Execution time: %.4s minutes ---' % ((time.time() - start_time) / 60)
