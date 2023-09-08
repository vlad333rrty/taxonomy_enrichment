import sys
import time

import pg
from nltk.corpus import wordnet as wn
from gensim.models.poincare import PoincareModel

from dao.PermissionType import PermissionType
from dao.dao_factory import DaoFactory
from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.essential_words_gathering_utils import gather_essential_words
from engines.processor_base import ProcessingResult
from engines.word_to_add_data import WordToAddDataParser
from result_printer import SemEvalTask2016FormatResultPrinter


def simple_name(r):
    return r.name().split('.')[0]


def append_pairs(root, words, pairs):
    stack = [root]
    used = set()
    while len(stack) > 0:
        current = stack.pop()
        if current.name() in used:
            continue
        for w in current.hyponyms():
            l = w.lemmas()
            r = current.lemmas()
            for ll in l:
                ln = ll.name()
                for rr in r:
                    rn = rr.name()
                    pairs.add((ln, rn))
                    words.add(ln)
                    words.add(rn)
            stack.append(w)
        used.add(current.name())


def append_pairs_upstairs(my_root, words):
    stack = [my_root]
    pairs = []
    while len(stack) > 0:
        current = stack.pop()
        for w in current.hyponyms():
            x = simple_name(w)
            y = simple_name(current)
            pairs.append((simple_name(my_root), simple_name(w)))
            words.add(x)
            words.add(y)
            stack.append(w)
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

def paginate(entries, page_size):
    result = []
    k = len(entries) // page_size
    for j in range(k):
        page = []
        for r in range(page_size):
            index = j * page_size + r
            page.append(entries[index])
        result.append(page)
    page = []
    i = k * page_size
    while i < len(entries):
        page.append(entries[i])
        i += 1
    result.append(page)
    return result

def get_poincare_embeddings_holder(relations_all):
    model = PoincareModel(relations_all, size=10, negative=min(len(relations_all), 10))
    model.train(epochs=20)
    return model.kv


def get_synset(similar_words, embeddings_holder):
    _res = None
    score = -1
    for similar_and_embed in similar_words:
        similar = similar_and_embed[0]
        synsets = wn.synsets(similar)
        if len(synsets) == 0:
            continue
        for synset in synsets:
            best_lemma_score = -1
            for lemma in synset.lemmas():
                name = lemma.name()
                if not embeddings_holder.has_index_for(name):
                    if embeddings_holder.has_index_for(name.lower()):
                        name = name.lower()
                    else:
                        continue
                cur_score = embeddings_holder.similarity(name, similar)
                if cur_score > best_lemma_score:
                    best_lemma_score = cur_score
            if best_lemma_score > score:
                score = best_lemma_score
                _res = synset
        if _res is not None:
            return _res
    return _res

def test():
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    relations_all, words = get_relations(words_to_add_data)
    embeddings_holder = get_poincare_embeddings_holder(relations_all)
    res = []
    for word2add_data in words_to_add_data:
        word = word2add_data.value
        try:
            similar_words = embeddings_holder.similar_by_key(word)
            synset = get_synset(similar_words, embeddings_holder)
            res.append(ProcessingResult(word, word2add_data.num, [synset], 'attach'))
        except Exception as e:
            print(e)
    SemEvalTask2016FormatResultPrinter('result/tmp.csv').print_result(res)
    print(res)

def get_relations(words_to_add_data):
    token2essential_words, num2word = gather_essential_words(words_to_add_data, 10)

    used_synsets = set()
    synsets_all = []
    for key in token2essential_words:
        essentials = token2essential_words[key][0]
        synsets = list(
            map(
                lambda e_w: wn.synsets(e_w),
                essentials
            )
        )
        synsets_fallten = [s for ss in synsets for s in ss]
        for synset in synsets_fallten:
            if synset.name() not in used_synsets:
                used_synsets.add(synset.name())
                synsets_all.append(synset)

    print('Gathering relations...')
    relations_all = set()
    words = set()
    i = 0
    for synset in synsets_all:
        print('Processing synset ', synset)
        append_pairs(synset, words, relations_all)

        i += 1
        print('Processed {}%'.format(i / len(synsets_all)))
    return relations_all, words


test()