t2s = set()
deduped = []
with open('data/results/TEMP/predicted.tsv', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        arr = line.split('\t')
        term = arr[0]
        s = arr[1]
        if (term, s) in t2s:
            deduped.append(term)
        else:
            t2s.add((term, s))


used_terms = set(
    map(
        lambda x: x[0],
        t2s
    )
)
with open('data/datasets/diachronic-wordnets/en/no_labels_nouns_en.2.0-3.0.tsv', 'r') as file:
    lines = file.readlines()
    print(len(lines))
    words = set()
    for line in lines:
        line = line.strip()
        if line not in used_terms:
            words.add(line)
print(len(words))
with open('data/datasets/diachronic-wordnets/en/no_labels_unprocessed', 'w') as file:
    s = ''
    for word in words:
        s += f'{word}\n'
    file.write(s)


