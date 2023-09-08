import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pg

from dao.word_embeddings_dao import WordEmbeddingsDao
import glob

from utils.similarity import cos_sim


# x = read_all_embeddings(glob.glob('poincare_embeddings/*'))


def __read_entries(path):
    entries = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split(' ')
            word = tokens[0]
            embedding = list(map(float, tokens[1:]))
            entries.append(WordEmbeddingsDao.Entry(word, embedding))
    return entries

class TestSet:
    def __init__(self):
        self.container = set()
        self.duplicates = set()

    def add(self, word):
        if word in self.container:
            self.duplicates.add(word)
        else:
            self.container.add(word)

def insert_embeddings(cached_embeddings_format_str, table_name):
    paths = glob.glob(cached_embeddings_format_str)
    paths.remove('embeddings/crawl-300d-2M.vec')
    id2status = [0 for _ in paths]
    def insert_chunk(path, i):
        db = pg.DB(dbname='postgres', host='localhost', user='Vlad', port=32768, passwd='123456')
        dao = WordEmbeddingsDao(db, table_name)
        entries = __read_entries(path)
        print('Read ', len(entries))
        try:
            dao.insert_many(entries)
        except Exception as e:
            print('Error for ', i, ' exception ', e)
            id2status[i] = 1
            return
        id2status[i] = 1
        print('Inserted for ', path, ' with n= ', i)

    executor = ThreadPoolExecutor()
    for i in range(len(paths)):
        executor.submit(lambda t = i: insert_chunk(paths[t], t))
    start = time.time()
    while sum(id2status) != len(id2status):
        if time.time() - start > 5:
            start = time.time()
            r = ''
            for i in range(len(id2status)):
                if id2status[i] == 0:
                    r += str(i) +' '
            log_str = '{}% completed\nWaiting for:\n{}'.format(sum(id2status) / len(id2status), r)
            print(log_str)


if __name__ == '__main__':
    insert_embeddings('embeddings/*', 'word_embeddings')