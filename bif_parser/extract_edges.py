import re
import json

def extract(file):
    file = open(file, "r")
    regex = re.compile(r'\((.*)\)')
    edges = []
    for line in file:
        if not line.startswith('probability'):
            continue
        edges_as_string = re.search(regex, line).group().strip('()').replace(',','|').split('|')
        all_edges = list(map(lambda x: x.strip(), edges_as_string))
        for i in all_edges[1:]:
            edges.append((all_edges[0],i))

    with open('edge.txt', 'w') as outfile:
        json.dump(edges, outfile)

extract('/Users/constantin/Documents/bn/asia.bif')
