import json

def main():
    n = 8
    with open('solution_as_edges.json') as file:
        solutions = json.load(file)

    original = set([f'{i[0]}{i[1]}' for i in solutions['original']])
    predicted = set([f'{i[0]}{i[1]}' for i in solutions['predicted']])

    confusion_matrix = {
        'tn':0,
        'tp':len(original.intersection(predicted)),
        'fp':len(predicted - original),
        'fn':len(original - predicted)
    }
    confusion_matrix['tn'] = n*(n-1)/2-confusion_matrix['tp']-confusion_matrix['fp']-confusion_matrix['fn']
    print(confusion_matrix)
