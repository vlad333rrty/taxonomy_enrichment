import nltk
from nltk.corpus.reader import Synset

from utils.embeddings_extractor import extract_embedding
from utils.similarity import cos_sim
from k_closest import k_nearest_async

import common.performance as perf

FASTTEXT_EMBEDDINGS_PATH = 'embeddings/crawl-300d-2M.vec'


def __find_synset_candidates(query_word_embedding):
    wordnet = nltk.corpus.wordnet31
    pos = 'n'
    ts, closest = perf\
        .measure(lambda: k_nearest_async(5, query_word_embedding, FASTTEXT_EMBEDDINGS_PATH, lambda x: len(wordnet.synsets(x)) > 0))
    print(ts)
    if closest is None:
        return None

    synset_candidates = list(map(wordnet.synsets, closest))
    synset_candidates = [item for sublist in synset_candidates for item in sublist] # flatten
    synset_candidates = filter(lambda s: s.pos() == pos, synset_candidates) # filter by POS

    hypernym_candidates = filter(lambda s: s.pos == pos, map(Synset.hypernyms, synset_candidates))
    final_candidates = list(synset_candidates) + list(hypernym_candidates)
    return final_candidates


def __flatten_list(_2dlist):
    return [item for sublist in _2dlist for item in sublist]


def __get_word2synsets(synset_candidates):
    word_to_synset = {}
    for candidate in synset_candidates:
        lemma_names = candidate.lemma_names()
        for lemma_name in lemma_names:
            if lemma_name in word_to_synset:
                word_to_synset[lemma_name].append(candidate)
            else:
                word_to_synset[lemma_name] = [candidate]

    return word_to_synset


def __filter_synset_duplicates(synsets):
    used_synsets = set()
    result_synsets = []
    for synset in synsets:
        if synset.name() not in used_synsets:
            result_synsets.append(synset)
    return result_synsets


def __extract_suitable_synsets(query_word_embedding, synset_candidates):
    word2synsets = __get_word2synsets(synset_candidates)
    lemmas_set = word2synsets.keys()
    word2score = {}
    for lemma in lemmas_set:
        lemma_embedding = extract_embedding(lemma, FASTTEXT_EMBEDDINGS_PATH)
        if lemma_embedding is None:
            word2score[lemma] = -1  # todo consider
        else:
            word2score[lemma] = cos_sim(query_word_embedding, lemma_embedding)

    synsets_and_scores = []
    for key in word2synsets:
        synsets = __filter_synset_duplicates(word2synsets[key])
        score = word2score[key]
        synsets_and_scores.append((synsets, score))

    return sorted(synsets_and_scores, key=lambda x: x[1], reverse=True)


def find_candidates(query_word: str):
    query_word_embedding = extract_embedding(query_word, FASTTEXT_EMBEDDINGS_PATH)
    candidates = __find_synset_candidates(query_word_embedding)
    return __extract_suitable_synsets(query_word_embedding, candidates)
