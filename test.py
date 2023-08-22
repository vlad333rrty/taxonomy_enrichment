import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pg

from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.poincare_embeddings_based_processor import read_embeddings_dict_in_memory, einstein_midpoint
import glob

from utils.similarity import cos_sim


# x = read_all_embeddings(glob.glob('poincare_embeddings/*'))

def normalize(vec):
    return np.array(vec) / np.linalg.norm(vec)


def generate_samples(samples_count, dimension):
    samples = []
    for i in range(samples_count):
        sample = None
        while sample is None or np.linalg.norm(sample) >= 1:
            sample = [random.random() for _ in range(dimension)]
        samples.append(sample)
    return samples

def test_einstein_midpoint():
    word = 'testX'
    fasttext_embeddings_mapping = {
        'testX': [0.7069229895158972, 0.06353377989847564, 0.19485741001474643],
        'testY': [0.5864122624187564, 0.38476761396114045, 0.09199912423927481],
        'testZ': [0.4672471119623438, 0.2383340528287603, 0.558976665377226],
        'abracadabra': [0.3408929243494122, 0.3914575412130955, 0.3149153860597699],
        'abcde': [0.3287233300026001, 0.5510378955206098, 0.2420308787117027],
        'xyz': [0.26523325953248833, 0.3910865105614386, 0.8110840925365713],
        'alpha': [0.006263894970515604, 0.6198176057950735, 0.5233728166648772],
        'Z': [0.11040636254548963, 0.7210525458194678, 0.02532227568757983]
    }

    res = einstein_midpoint(word, fasttext_embeddings_mapping, 3)

    assert np.linalg.norm(res) < 1
    print(np.linalg.norm(res))
    print(res)


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

def insert_embeddings(cached_embeddings_format_str, table_name):
    paths = glob.glob(cached_embeddings_format_str)
    id2status = [0 for _ in paths]
    def insert_chunk(path, i):
        print('Start for ', i)
        db = pg.DB(dbname='postgres', host='localhost', user='Vlad', port=32768,
                   passwd='123456')
        dao = WordEmbeddingsDao(db, table_name)
        entries = __read_entries(path)
        try:
            dao.insert_many(entries)
        except Exception as e:
            print('Error for ', i)
            id2status[i] = 1
            return
        id2status[i] = 1
        print('Inserted for ', i)
    executor = ThreadPoolExecutor()
    i = 0
    for path in paths:
        executor.submit(lambda t=i: insert_chunk(path, t))
        i += 1
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
    pass
    # insert_embeddings('embeddings/*', 'word_embeddings')