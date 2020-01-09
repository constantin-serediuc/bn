
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_edge(1,2)
G.add_edge(1,3)
print(G.edges) # in

nx.draw(G, with_labels=True)
plt.draw()
plt.show()