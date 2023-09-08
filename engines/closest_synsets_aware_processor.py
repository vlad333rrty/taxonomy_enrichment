import time

from nltk.corpus.reader import Synset

from common import performance
from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.processor_base import Processor, ProcessingResult
from engines.word_to_add_data import WordToAddData
from utils.similarity import cos_sim
from nltk.corpus import wordnet31 as wn


class ClosestSynsetsAwareProcessor(Processor):

    def __init__(self, dao: WordEmbeddingsDao):
        self.__dao = dao

    def process(self, words_to_add_data: [WordToAddData]):
        start = time.time()
        result = []
        ts, word2embedding = performance.measure(self.__dao.find_all_as_map)
        print('Got embeddings in ', ts)
        counter = 0
        entries_with_scores = self.__get_entries_sorted_by_angle(word2embedding)
        for word_to_add_data in words_to_add_data:
            word = word_to_add_data.value
            if word not in word2embedding:
                print('No embedding found for {}'.format(word))
                return
            closest = self.__get_closest_word_bin(word, word2embedding[word], entries_with_scores)
            closest_synsets = self._get_k_nearest_synsets(word, closest, word2embedding, 5)
            print('Closest word for {} is {}'.format(word, closest))
            result.append(ProcessingResult(word, word_to_add_data.num, closest_synsets, 'attach'))
            counter += 1
            print('{} processed'.format(counter / len(words_to_add_data)))
        print('Finished in ', time.time() - start)
        return result

    def _get_k_nearest_synsets(self, word, closest_word_candidate, word2embedding, k):
        synsets = wn.synsets(closest_word_candidate)
        all_lemmas = list(map(Synset.lemma_names, synsets))
        lemmas_and_synset = zip(all_lemmas, synsets)

        if word not in word2embedding:
            return synsets[:k]

        word_embedding = word2embedding[word]

        score_and_synsets = map(lambda x: (
            max(map(
                lambda y: cos_sim(word_embedding,
                                  word2embedding[y]) if y in word2embedding else -1,
                x[0])), x[1]), lemmas_and_synset)

        return list(map(lambda x: x[1], sorted(score_and_synsets, key=lambda x: x[0], reverse=True)))[:k]


    def __get_sort_axis(self, dimension):
        ox = [0 for _ in range(dimension)]
        ox[0] = 1
        return ox

    def __get_entries_sorted_by_angle(self, word2embedding):

        ox = None
        entries_with_scores = []
        for key in word2embedding:
            embedding = word2embedding[key]
            if ox is None:
                ox = self.__get_sort_axis(len(embedding))
            entries_with_scores.append((key, embedding, cos_sim(ox, embedding)))
        return sorted(entries_with_scores, key=lambda triple: triple[2])

    def __get_closest_word_bin(self, word, embedding, entries_with_scores):
        size = len(entries_with_scores)
        ox = self.__get_sort_axis(len(embedding))
        sim = cos_sim(embedding, ox)
        l, r = -1, size
        while l != r - 1:
            m = (r - l) // 2 + l
            if entries_with_scores[m][2] > sim:
                r = m
            else:
                l = m
        left_bound, right_bound = l, r
        while left_bound >= 0 and entries_with_scores[left_bound] == sim:
            left_bound -= 1
        left_bound = max(left_bound, 0)
        while right_bound < size and entries_with_scores[right_bound] == sim:
            right_bound += 1
        right_bound = min(right_bound, size - 1)
        best_score = -1
        most_similar = None
        for i in range(left_bound, right_bound + 1):
            current_word = entries_with_scores[i][0]
            if current_word == word:
                continue
            current_sim = cos_sim(embedding, entries_with_scores[i][1])
            if current_sim > best_score:
                best_score = current_sim
                most_similar = current_word
        return most_similar

    @staticmethod
    def __get_closest_word(word, word2embedding):
        """
        todo sort by cos sim and find proper vector somehow?
        :param word:
        :param word2embedding:
        :return:
        """
        closest = None
        best_score = -1
        embedding = word2embedding[word]
        for key in word2embedding:
            if key == word:
                continue
            sample = word2embedding[key]
            sim = cos_sim(embedding, sample)
            if sim > best_score:
                best_score = sim
                closest = key
        return closest
