import networkx as nx
from networkx import grid_graph

import matplotlib.pyplot as plt

# G=nx.complete_graph(['hi', 'my','name','is','Richard'])
# G.edges()
G=grid_graph(dim=[2,3])
# G = nx.Graph()
#
# G.add_node("START")

nx.draw(G, with_labels=True, font_weight='bold')

plt.show()
