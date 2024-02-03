from multiprocessing.pool import ThreadPool

from src.taxo_expantion_methods.common.performance import measure


def __read_data(path: str, seek: int, limit: int):
    res = {}
    with open(path, 'r') as file:
        file.seek(seek)
        for _ in range(limit):
            line = file.readline().split(' ')
            word = line[0]
            embedding = list(map(float, line[1:]))
            res[word] = embedding
        return res, file.tell()


def __read_file_full(path: str):
    res = {}
    with open(path, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            line = line.split(' ')
            word = line[0]
            embedding = list(map(float, line[1:]))
            res[word] = embedding
        return res


def __find_new_closest_and_update_result_vector(word_to_embedding, etalon, predicate, scores, result):
    for key in word_to_embedding:
        if not predicate(key):
            continue
        embedding = word_to_embedding[key]
        cos = src.utils.similarity.cos_sim(embedding, etalon)
        if cos < scores[-1]:
            continue
        for i in range(len(scores)):
            if scores[i] < cos:
                for j in range(len(scores) - 1, i, -1):
                    scores[j] = scores[j - 1]
                    result[j] = result[j - 1]
                scores[i] = cos
                result[i] = key
                break


def k_closest(k: int, etalon_embedding, data_path, predicate):
    if etalon_embedding is None:
        return None
    total = 0
    pivot = 12
    limit = 10000
    result = ['' for _ in range(k)]
    scores = [-1 for _ in range(k)]
    while total < 2 * 1e+6:
        res, pivot = __read_data(data_path, pivot, limit)
        __find_new_closest_and_update_result_vector(res, etalon_embedding, predicate, scores, result)
        total += limit
    return result


def k_nearest(k: int, etalon_embedding, data_path, predicate):
    if etalon_embedding is None:
        return None
    result = ['' for _ in range(k)]
    scores = [-1 for _ in range(k)]
    res, pivot = __read_file_full(data_path)
    __find_new_closest_and_update_result_vector(res, etalon_embedding, predicate, scores, result)
    return result


def __read_file_chunk(path: str, start: int, end: int):
    res = {}
    with open(path, 'r') as file:
        lines = file.readlines()[start:end]
        for line in lines:
            line = line.split(' ')
            word = line[0]
            embedding = list(map(float, line[1:]))
            res[word] = embedding
        return res


def k_nearest_inner(k: int, etalon_embedding, data_path, predicate, x):
    if etalon_embedding is None:
        return None
    result = ['' for _ in range(k)]
    scores = [-1 for _ in range(k)]
    ts, res = measure(lambda: __read_file_full(data_path))
    print('reading file \'{}\' : {}'.format(data_path, ts))
    ts, _ = measure(lambda: __find_new_closest_and_update_result_vector(res, etalon_embedding, predicate, scores, result))
    print('Processing: {}'.format(ts))
    x.append(1)
    print('Completed: {}'.format(len(x) / 1000))
    return result, scores


# 1999996

def k_nearest_async(k: int, etalon_embedding, data_path, predicate):
    x = []

    pool = ThreadPool(100)
    pool.map(lambda path: k_nearest_inner(k, etalon_embedding, path, predicate, x), [data_path + str(i) for i in range(1000)])
    pool.close()
    pool.join()


