
model = 'TEMP'
with open('data/results/{}/predicted_food.tsv'.format(model), 'r') as file:
    lines = file.readlines()
    res = []
    i = 1
    for line in lines:
        line = line.strip()
        x = line.split('\t')
        if len(x) != 2:
            print('Len < 2', line)
            i += 1
            continue
        word = x[0]
        terms = x[1]
        res.append('{}\t{}'.format(word, terms))
        i += 1

with open('data/results/semeval/{}/predicted2.tsv'.format(model), 'w') as file:
    file.write('\n'.join(res))

