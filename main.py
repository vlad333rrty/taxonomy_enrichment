from nltk.corpus import wordnet31 as wn
x = wn.synset('beginning.n.02')

print(x.definition())

y = wn.synset('opening.n.02')

gold_lemmas = set(str(lemma.name()) for lemma in x.lemmas())
system_lemmas = set(str(lemma.name()) for lemma in y.lemmas())

for l in gold_lemmas:
    if l in system_lemmas:
        print('yes')
