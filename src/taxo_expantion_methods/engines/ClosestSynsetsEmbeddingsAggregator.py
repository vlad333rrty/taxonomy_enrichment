from abc import ABC, abstractmethod

import numpy as np

from src.taxo_expantion_methods.dao.word_embeddings_dao import WordEmbeddingsDao


class ClosestSynsetsEmbeddingsAggregator(ABC):

    def __init__(self, dao: WordEmbeddingsDao):
        self.__dao = dao

    def aggregate_and_insert_into_table(self, word, synsets):
        print('Dump poincare embedding')

        approx_points = []
        for s in synsets:
            lemmas = s.lemmas()
            for lemma in lemmas:
                name = lemma.name()
                embedding = self.__dao.find_or_none(name)
                if embedding is not None:
                    approx_points.append(embedding)
                    break
        approx_points = approx_points[:5]
        if len(approx_points) > 0:
            approx = self._average(approx_points)
            try:
                self.__dao.insert_many([WordEmbeddingsDao.Entry(word, [x for x in approx])])
            except Exception as e:
                print(e)
            pass
        else:
            print('No approx points found for ', word)

    @abstractmethod
    def _average(self, points):
        raise NotImplementedError()


class PoincareClosestSynsetsEmbeddingsAggregator(ClosestSynsetsEmbeddingsAggregator):

    @staticmethod
    def __einstein_midpoint(k_nearest_embeddings) -> np.array:
        nominator = 0
        denominator = 0
        for x in k_nearest_embeddings:
            l = 1 / np.sqrt(1 - np.linalg.norm(x) ** 2)
            nominator += l * np.array(x)
            denominator += l
        return nominator / denominator

    def _average(self, points):
        return self.__einstein_midpoint(points)


class AveragingClosestSynsetsEmbeddingsAggregator(ClosestSynsetsEmbeddingsAggregator):
    def _average(self, points):
        np_points = list(
            map(
                lambda l: np.array(l),
                points
            )
        )
        return sum(np_points) / len(np_points)
