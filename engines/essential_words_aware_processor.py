from abc import ABC, abstractmethod

import nltk
import numpy as np
from nltk.corpus.reader import Synset

from common.lemmatizer import Lemmatizer
from dao.dao_factory import DaoFactory
from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.essential_words_gathering_utils import gather_essential_words
from engines.processor_base import Processor, ProcessingResult
from engines.word_to_add_data import WordToAddData
from utils import utils
from utils.similarity import cos_sim
from common.performance import measure
from utils.simple_singulizer import get_singular


class EssentialWordsAwareProcessor(Processor, ABC):

    def __init__(self):
        self.__lemmatizer = Lemmatizer()
        self.__closest_sysnets_embeddings_aggregator = None

    def set_closest_sysnets_embeddings_aggregator(self, closest_sysnets_embeddings_aggregator):
        self.__closest_sysnets_embeddings_aggregator = closest_sysnets_embeddings_aggregator

    @abstractmethod
    def _gather_embeddings_for_words(self, holder) -> dict:
        raise NotImplementedError()

    def process(self, words_to_add_data: [WordToAddData]):
        result = []
        token2essential_words, num2word = gather_essential_words(words_to_add_data, 10)

        query_words = []
        for key in token2essential_words:
            query_words.append(num2word[key])
            query_words += token2essential_words[key][0]

        ts, word_to_embedding = measure(
            lambda: self._gather_embeddings_for_words(query_words))
        print('Gathered embeddings in {}s'.format(ts))

        counter = 0
        for word_to_add_token in token2essential_words:
            word = num2word[word_to_add_token]
            essential_words = token2essential_words[word_to_add_token][0]
            pos = self.__get_appropriate_pos(token2essential_words[word_to_add_token][1])
            if len(essential_words) > 0:
                lemmas = list(
                    map(
                        lambda e: get_singular(self.__lemmatizer.get_words_normal_form(e)),
                        essential_words
                    )
                )
                if word_to_add_token in ['withdef.160', 'withdef.247', 'withdef.238', 'withdef.322']:
                    print(1)
                k_nearest = self._get_k_nearest_words(word, lemmas, word_to_embedding, 10)
                synsets = self._get_nearest_words_synsets_filtered_by_pos(k_nearest, pos)
                if len(synsets) == 0:
                    print('No synsets found for {}'.format(word_to_add_token))
                    continue
                closest_synsets = self._get_k_nearest_synsets(word_to_add_token, synsets, word_to_embedding, 5)
                if self.__closest_sysnets_embeddings_aggregator is not None:
                    self.__closest_sysnets_embeddings_aggregator.aggregate_and_insert_into_table(word, closest_synsets)

                result.append(ProcessingResult(word, word_to_add_token, closest_synsets, 'attach'))
            counter += 1
            print('{}% completed'.format(counter / len(token2essential_words)))

        return result

    def __get_appropriate_pos(self, pos: str):
        return 'n' if pos == 'noun' else 'v'

    def _get_k_nearest_words(self, word: str, essential_words, word_to_embedding, k: int):
        if word not in word_to_embedding:
            print('No embedding found in cache for ', word)
            return essential_words[:k]  # todo hack or heuristic?

        embedding = word_to_embedding[word]

        words_and_score = filter(
            lambda x: x[1] is not None,
            map(
                lambda w: (w, cos_sim(embedding, word_to_embedding[w]) if w in word_to_embedding else None),
                essential_words
            )
        )
        return list(map(lambda x: x[0], sorted(words_and_score, key=lambda x: x[1], reverse=True)))[:k]

    def _get_nearest_words_synsets_filtered_by_pos(self, k_nearest, pos):
        wordnet = nltk.corpus.wordnet31
        synsets = list(map(wordnet.synsets, k_nearest))
        filtered = list(filter(lambda x: x.pos() == pos, [item for synset in synsets for item in synset]))
        if len(filtered) > 0:
            return filtered
        synsets_flat = [item for synset in synsets for item in synset]
        synsets_extended = utils.get_extended_synset_list(synsets_flat)
        filtered = list(filter(lambda x: x.pos() == pos, [item for synset in synsets_extended for item in synset]))
        return filtered


    def _get_k_nearest_synsets(self, word, synsets, word_to_embedding, k):
        all_lemmas = list(map(Synset.lemma_names, synsets))
        lemmas_and_synset = zip(all_lemmas, synsets)

        if word not in word_to_embedding:
            return synsets[:k]

        word_embedding = word_to_embedding[word]

        score_and_synsets = map(lambda x: (
            max(map(
                lambda y: cos_sim(word_embedding,
                                  word_to_embedding[y]) if y in word_to_embedding else -1,
                x[0])), x[1]), lemmas_and_synset)

        return list(map(lambda x: x[1], sorted(score_and_synsets, key=lambda x: x[0], reverse=True)))[:k]


class EmbeddingsBasedEssentialWordsAwareProcessor(EssentialWordsAwareProcessor):
    def __init__(self, embeddings_dao: WordEmbeddingsDao):
        super().__init__()
        self.__embeddings_dao = embeddings_dao

    def _gather_embeddings_for_words(self, query_words) -> dict:
        return self.__embeddings_dao.find_by_keys_as_map(query_words)


class PoincareEmbeddingsEssentialWordsAwareProcessor(EssentialWordsAwareProcessor):
    def __init__(self, poincare_embeddings_dao: WordEmbeddingsDao):
        super().__init__()
        self.__poincare_embeddings_dao = poincare_embeddings_dao

    def _gather_embeddings_for_words(self, query_words) -> dict:
        word2poincare_embedding = self.__poincare_embeddings_dao.find_by_keys_as_map(query_words)
        absent_words = list(
            filter(
                lambda w: w not in word2poincare_embedding,
                query_words
            )
        )
        if len(absent_words) > 0:
            print('Warning, no poincare embeddings found for ', absent_words)

        return word2poincare_embedding