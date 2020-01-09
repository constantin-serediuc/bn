import re
import json

def extract(file):
    """
    save to edge.txt file a list of edges in (parent,child) format
    """
    file = open(file, "r")
    regex = re.compile(r'\((.*)\)')
    edges = []
    for line in file:
        if not line.startswith('probability'):
            continue
        edges_as_string = re.search(regex, line).group().strip('()').replace(',','|').split('|')
        all_edges = list(map(lambda x: x.strip(), edges_as_string))
        for i in all_edges[1:]:
            edges.append((i,all_edges[0]))

    with open('edge.txt', 'w') as outfile:
        json.dump(edges, outfile)

extract('../asia.bif')
