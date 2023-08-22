import time

from nltk.corpus import wordnet as wn
from gensim.models.poincare import PoincareModel


def simple_name(r):
    return r.name().split('.')[0]


def append_pairs(my_root, pairs, words):
    for w in my_root.hyponyms():
        x = simple_name(w)
        y =simple_name(my_root)
        pairs.append((simple_name(w), simple_name(my_root)))
        words.add(x)
        words.add(y)
        append_pairs(w, pairs, words)
    return pairs


def dump_pairs(pairs, path):
    lines = []
    for pair in pairs:
        pojo = [pair[0]] + list(map(str, pair[1]))
        lines.append(' '.join(pojo) + '\n')
    with open(path, 'w') as file:
        file.writelines(lines)


def paginate_and_dump(pairs, page_size):
    bound_ceil = (len(pairs) + page_size - 1) // page_size
    for i in range(bound_ceil):
        page = pairs[i * page_size : (i + 1) * page_size]
        dump_pairs(page, 'poincare_embeddings/embeddings_{}'.format(i))

if __name__ == '__main__':
    start = time.time()
    root = wn.synset('entity.n.01')
    words = set()

    relations = append_pairs(root, [], words)

    model = PoincareModel(relations, size=300)
    model.train(epochs=20)

    pairs = list(
        map(
            lambda x: (x, model.kv.get_vector(x)),
            words
        )
    )
    paginate_and_dump(pairs, 5000)
