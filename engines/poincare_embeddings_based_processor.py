import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pg

import common.performance
from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.essential_words_gathering_utils import gather_essential_words
from engines.word_to_add_data import WordToAddData
from utils import embeddings_extractor
from utils.similarity import cos_sim


def read_embeddings_dict_in_memory(path: str):
    word2embedding = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split(' ')
            word = tokens[0]
            embedding = list(map(float, tokens[1:]))
            word2embedding[word] = embedding
    return word2embedding


def read_all_embeddings(paths: [str]):
    executor = ThreadPoolExecutor()
    futures = []
    for path in paths:
        future = executor.submit(lambda: read_embeddings_dict_in_memory(path))
        futures.append(future)
    result_dict = {}
    for future in futures:
        result_dict = {**result_dict, **future.result()}
    return result_dict

def shift_right(array, i):
    prev = array[i]
    for j in range(i + 1, len(array)):
        t = array[j]
        array[j] = prev
        prev = t

def get_k_nearest(word: str, word2fasttext_embedding: dict, k: int):
    word_embedding = word2fasttext_embedding[word]
    accum = [0 for _ in range(k)]
    result = [[] for _ in range(k)]
    for key in word2fasttext_embedding:
        embedding = word2fasttext_embedding[key]
        sim = cos_sim(word_embedding, embedding)
        for i in range(k):
            if accum[i] < sim:
                shift_right(accum, i)
                accum[i] = sim
                result[i] = embedding
                break
    return result


def einstein_midpoint(word: str, word2fasttext_embedding: dict, k: int):
    k_nearest = get_k_nearest(word, word2fasttext_embedding, k)

    nominator = 0
    denominator = 0
    for x in k_nearest:
        l = 1 / np.sqrt(1 - np.linalg.norm(x) ** 2)
        nominator += l * np.array(x)
        denominator += l
    return nominator / denominator

class PoincareEmbeddingsBasedProcessor:
    def __init__(self, word_embeddings_dao: WordEmbeddingsDao):
        self.__word_embeddings_dao = word_embeddings_dao

    def __get_poincare_embeddings(self, words_to_add_data: [WordToAddData], word2fasttext_embedding: dict):
        word2poincare_embedding = {}
        for word_to_add in words_to_add_data:
            word = word_to_add.value.lower()
            poincare_embedding = einstein_midpoint(word, word2fasttext_embedding, 5)
            word2poincare_embedding[word] = poincare_embedding
        return word2poincare_embedding

    def process(self, words_to_add_data: [WordToAddData]):
        delta, word2fasttext_embedding = common.performance.measure(lambda: self.__word_embeddings_dao.find_all_as_map())
        print('Got embeddings in {}'.format(delta))
        delta, word2poincare_embedding = common.performance.measure(lambda: self.__get_poincare_embeddings(words_to_add_data, word2fasttext_embedding))
        print('Gor poincare embeddings in {}'.format(delta))
        essential_words_per_word, word_to_num = gather_essential_words(words_to_add_data, 10)


        