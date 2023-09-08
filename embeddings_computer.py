import networkx as nx
from gensim.models import Word2Vec
from gensim.models.poincare import PoincareModel
from matplotlib import pyplot as plt
from node2vec import Node2Vec

from dao.PermissionType import PermissionType
from dao.dao_factory import DaoFactory
from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.essential_words_aware_processor import EmbeddingsBasedEssentialWordsAwareProcessor
from engines.essential_words_gathering_utils import gather_essential_words
from nltk.corpus import wordnet as wn

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
            l = simple_name(w)
            r = simple_name(current)
            pairs.add((l, r))
            words.add(l)
            words.add(r)
            stack.append(w)
        used.add(current.name())


def gather_all_synsets(words_to_add_data):
    token2essential_words, num2word = gather_essential_words(words_to_add_data, 2)

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

def gather_relations(synsets_all):

    print('Gathering relations...')
    relations_all = set()
    words = set()
    i = 0
    for synset in synsets_all:
        print('Processing synset ', synset)
        append_pairs(synset, words, relations_all)

        i += 1
        print('Processed {}%'.format(i / len(synsets_all)))


    print('Got {} words and {} relations'.format(len(words), len(relations_all)))
    return words, relations_all

# todo bad method
def compute_node_embeddings_as_entries(words, relations):
    g = nx.Graph()
    for word in words:
        g.add_node(word)
    for relation in relations:
        g.add_edge(relation[0], relation[1])

    print('Fitting...')
    node2vec = Node2Vec(g, dimensions=300, walk_length=30, num_walks=200, workers=8)
    model = node2vec.fit()

    def to_arr(a):
        return [x for x in a]

    entries = list(
        map(
            lambda x: WordEmbeddingsDao.Entry(x, to_arr(model.wv.get_vector(x))),
            words
        )
    )
    print('Got {} entries'.format(len(entries)))

    return entries


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


class LostDataSaver:
    i = 0
    def save(self, data: [WordEmbeddingsDao.Entry]):
        with open('lost_page_' + str(self.i), 'w') as file:
            lines = []
            for x in data:
                s = x.word + ' ' + ' '.join(map(str, x.embedding)) + '\n'
                lines.append(s)
            file.writelines(lines)
        self.i += 1

lost_page_saver = LostDataSaver()
def dump_to_dao(dao, entries):
    pages = paginate(entries, 10_000)
    for page in pages:
        try:
            dao.insert_many(page)
        except Exception as e:
            print(e)
            print('Serilize lost data into file...')
            lost_page_saver.save(page)


def run():
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    fasttext_dao = DaoFactory.create_word_embeddings_dao()
    a = ['withdef.9', 'withdef.63', 'withdef.89', 'withdef.160', 'withdef.188', 'withdef.232', 'withdef.238', 'withdef.240', 'withdef.246', 'withdef.254', 'withdef.263', 'withdef.268', 'withdef.270', 'withdef.271', 'withdef.277', 'withdef.299', 'withdef.302', 'withdef.322']
    process_result = EmbeddingsBasedEssentialWordsAwareProcessor(fasttext_dao).process(words_to_add_data)
    synsets2d = list(
        map(
            lambda r: r.candidate_synsets,
            filter(
                lambda x: x.word_token in a,
                process_result
            )
        )
    )
    synsets_flat = [s for synsets in synsets2d for s in synsets]
    print(len(synsets_flat))

    words, relations = gather_relations(synsets_flat)
    entries = compute_node_embeddings_as_entries(words, relations)
    dao = DaoFactory.create_node_embeddings_dao()
    entries = list(
        filter(
            lambda e: dao.find_or_none(e.word) is None,
            entries
        )
    )

    dump_to_dao(dao, entries)
    print(entries)


def get_embeddings_holder(words_to_add_data):
    fasttext_dao = DaoFactory.create_word_embeddings_dao()
    a = ['withdef.9']
    process_result = EmbeddingsBasedEssentialWordsAwareProcessor(fasttext_dao).process(words_to_add_data)
    synsets2d = list(
        map(
            lambda r: r.candidate_synsets,
            filter(
                lambda x: x.word_token in a,
                process_result
            )
        )
    )
    synsets_flat = [s for synsets in synsets2d for s in synsets]
    print(len(synsets_flat))

    words, relations = gather_relations(synsets_flat)

    g = nx.Graph()
    for word in words:
        g.add_node(word)
    for relation in relations:
        g.add_edge(relation[0], relation[1])

    print('Fitting...')
    node2vec = Node2Vec(g, dimensions=300, walk_length=30, num_walks=200, workers=8)
    model = node2vec.fit()
    return model.wv


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

    embeddings_holder = get_embeddings_holder(words_to_add_data)
    dao = DaoFactory.create_node_embeddings_dao(PermissionType.READ_ONLY)
    word2embedding = dao.find_all_as_map()
    keys = []
    vectors = []

    for key in word2embedding:
        keys.append(key)
        vectors.append(word2embedding[key])

    page_size = 10_000
    key_pages = paginate(keys, page_size)
    vector_pages = paginate(vectors, page_size)
    for i in range(len(key_pages)):
        embeddings_holder.add_vectors(key_pages[i], vector_pages[i])

    res = []
    for word2add_data in words_to_add_data:
        word = word2add_data.value
        try:
            if word == 'Gene Therapy':
                print(word)
            similar_words = embeddings_holder.similar_by_key(word)
            synset = get_synset(similar_words, embeddings_holder)
            if synset is None:
                print(synset)
            res.append(ProcessingResult(word, word2add_data.num, [synset], 'attach'))
        except Exception as e:
            print(e)
    SemEvalTask2016FormatResultPrinter('result/tmp.csv').print_result(res)
    print(res)



test()