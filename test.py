import networkx as nx
import numpy as np

G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('B', 'A')])
print(list(nx.simple_cycles(G)))
