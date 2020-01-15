import networkx as nx
import numpy as np

G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('B', 'C')])
print(nx.to_numpy_array(G))
print(np.count_nonzero(nx.to_numpy_array(G)))