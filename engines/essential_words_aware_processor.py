from abc import ABC, abstractmethod

import nltk
import numpy as np
from nltk.corpus.reader import Synset

from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.essential_words_gathering_utils import gather_essential_words
from engines.word_to_add_data import WordToAddDataParser, WordToAddData
from utils.embeddings_extractor import extract_embeddings
from utils.similarity import cos_sim
from common.performance import measure


class EssentialWordsAwareProcessor(ABC):
    @abstractmethod
    def _gather_embeddings_for_words(self, essential_words_per_word) -> dict:
        raise NotImplementedError()

    def process(self, words_to_add_data: [WordToAddData]):
        result = []
        essential_words_per_word, word_to_num = gather_essential_words(words_to_add_data, 10)

        ts, word_to_embedding = measure(
            lambda: self._gather_embeddings_for_words(essential_words_per_word))
        print('Gathered embeddings in {}s'.format(ts))

        counter = 0
        for word_to_add in essential_words_per_word:
            essential_words = essential_words_per_word[word_to_add][0]
            pos = self.__get_appropriate_pos(essential_words_per_word[word_to_add][1])
            if len(essential_words) > 0:
                k_nearest = self.__get_k_nearest_words(word_to_add, essential_words, word_to_embedding, 5)
                synsets = self.__get_nearest_words_synsets_filtered_by_pos(k_nearest, pos)
                if len(synsets) == 0:
                    print('No synsets found for {}'.format(word_to_add))
                    continue
                closest_synsets = self.__get_k_nearest_synsets(word_to_add, synsets, word_to_embedding, 5)
                result.append([
                    word_to_num[word_to_add],
                    closest_synsets[0],
                    'attach'  # todo temp
                ])
            counter += 1
            print('{}% completed'.format(counter / len(essential_words_per_word)))

        return result

    def __get_appropriate_pos(self, pos: str):
        return 'n' if pos == 'noun' else 'v'

    def __get_k_nearest_words(self, word: str, essential_words, word_to_embedding, k: int):
        if word.lower() not in word_to_embedding:
            return essential_words[:k]  # todo hack or heuristic?

        embedding = word_to_embedding[word.lower()]

        essential_words_embeddings = filter(lambda x: x is not None,
                                            map(lambda w: word_to_embedding[
                                                w.lower()] if w.lower() in word_to_embedding else None,
                                                essential_words))
        words_and_score = map(lambda p: (cos_sim(embedding, p[0]), p[1]),
                              zip(essential_words_embeddings, essential_words))
        return list(map(lambda x: x[1], sorted(words_and_score, key=lambda x: x[0], reverse=True)))[:k]

    def __get_nearest_words_synsets_filtered_by_pos(self, k_nearest, pos):
        wordnet = nltk.corpus.wordnet31
        synsets = map(wordnet.synsets, k_nearest)
        return list(filter(lambda x: x.pos() == pos, [item for synset in synsets for item in synset]))

    def __get_k_nearest_synsets(self, word, synsets, word_to_embedding, k):
        all_lemmas = list(map(Synset.lemma_names, synsets))
        lemmas_and_synset = zip(all_lemmas, synsets)

        # unknown_lemmas = self.__get_all_unknown_lemmas(word_to_embedding,
        #                                           [lemma for lemmas in all_lemmas for lemma in lemmas])
        # if len(unknown_lemmas) > 0:
        #     print('Going to update cache...')
        #     ts, _ = measure(lambda: __update_cache(word_to_embedding, unknown_lemmas, embeddings_path))
        #     print('Cache updated in {}s'.format(ts))

        if word.lower() not in word_to_embedding:
            return synsets[:k]

        word_embedding = word_to_embedding[word.lower()]

        score_and_synsets = map(lambda x: (
            max(map(
                lambda y: cos_sim(word_embedding,
                                  word_to_embedding[y.lower()]) if y.lower() in word_to_embedding else -1,
                x[0])), x[1]),
                                lemmas_and_synset)

        return list(map(lambda x: x[1], sorted(score_and_synsets, key=lambda x: x[0], reverse=True)))[:k]

    def __get_all_unknown_lemmas(self, word_to_embedding, lemmas):
        return list(filter(lambda l: l not in word_to_embedding, lemmas))


class FasttextEssentialWordsAwareProcessor(EssentialWordsAwareProcessor):
    def __init__(self, fasttext_dao: WordEmbeddingsDao):
        self.__fasttext_dao = fasttext_dao

    def _gather_embeddings_for_words(self, essential_words_per_word) -> dict:
        query_words = []
        for key in essential_words_per_word:
            query_words.append(key)
            query_words += essential_words_per_word[key][0]
        return self.__fasttext_dao.find_embeddings_by_keys(query_words)


class PoincareEmbeddingsEssentialWordsAwareProcessor(EssentialWordsAwareProcessor):
    def __init__(self, fasttext_dao: WordEmbeddingsDao, poincare_embeddings_dao: WordEmbeddingsDao):
        self.__fasttext_dao = fasttext_dao
        self.__poincare_embeddings_dao = poincare_embeddings_dao

    def _gather_embeddings_for_words(self, essential_words_per_word) -> dict:
        query_words = []
        for key in essential_words_per_word:
            query_words.append(key)
            query_words += essential_words_per_word[key][0]
        word2poincare_embedding = self.__poincare_embeddings_dao.find_embeddings_by_keys(query_words)
        absent_words = list(
            filter(
                lambda w: w not in word2poincare_embedding,
                essential_words_per_word.keys()
            )
        )
        if len(absent_words) > 0:
            print('Going to calculate poincare embeddings for ', absent_words, '\nsize: ', len(absent_words))

            def calculate_embeddings_closure():
                word2fasttext_embedding = self.__fasttext_dao.find_all_as_map()
                return PoincareEmbeddingsEssentialWordsAwareProcessor.__calculate_poincare_embeddings(absent_words,
                                                                                                      word2fasttext_embedding)

            delta, calculated_embeddings = measure(calculate_embeddings_closure)
            print('Poincare embeddings calculation finished in ', delta)
            print('Dumping these embeddings...')
            delta, _ = measure(lambda: self.__dump_poincare_embeddings(calculated_embeddings))
            print('Embeddings inserted in ', delta)
            return {**word2poincare_embedding, **calculated_embeddings}
        return word2poincare_embedding

    def __dump_poincare_embeddings(self, calculated_embeddings_dict: dict):
        entries = WordEmbeddingsDao.Entry.from_dict(calculated_embeddings_dict)
        self.__poincare_embeddings_dao.insert_many(entries)

    @staticmethod
    def __calculate_poincare_embeddings(words_to_add_data: [str], word2fasttext_embedding: dict):
        word2poincare_embedding = {}
        for word_to_add in words_to_add_data:
            word = word_to_add.lower()
            poincare_embedding = (PoincareEmbeddingsEssentialWordsAwareProcessor
                                  .__einstein_midpoint(word, word2fasttext_embedding, 5))
            word2poincare_embedding[word] = poincare_embedding
        return word2poincare_embedding

    @staticmethod
    def __shift_right(array, i):
        prev = array[i]
        for j in range(i + 1, len(array)):
            t = array[j]
            array[j] = prev
            prev = t

    @staticmethod
    def __get_k_nearest(word: str, word2fasttext_embedding: dict, k: int):
        word_embedding = word2fasttext_embedding[word]
        accum = [0 for _ in range(k)]
        result = [[] for _ in range(k)]
        for key in word2fasttext_embedding:
            embedding = word2fasttext_embedding[key]
            sim = cos_sim(word_embedding, embedding)
            for i in range(k):
                if accum[i] < sim:
                    PoincareEmbeddingsEssentialWordsAwareProcessor.__shift_right(accum, i)
                    accum[i] = sim
                    result[i] = embedding
                    break
        return result

    @staticmethod
    def __einstein_midpoint(word: str, word2fasttext_embedding: dict, k: int):
        k_nearest = PoincareEmbeddingsEssentialWordsAwareProcessor.__get_k_nearest(word, word2fasttext_embedding, k)

        nominator = 0
        denominator = 0
        for x in k_nearest:
            l = 1 / np.sqrt(1 - np.linalg.norm(x) ** 2)
            nominator += l * np.array(x)
            denominator += l
        return nominator / denominator
