import random
import time

from nltk.corpus.reader import Synset


class PathsBatch:
    def __init__(self, positive_paths, negative_paths):
        self.positive_paths = positive_paths
        self.negative_paths = negative_paths


class TEMPDsCreator:
    def __init__(self, all_synsets, negative_samples_per_node=1):
        self.__all_synsets = all_synsets
        self.__negative_samples_per_node = negative_samples_per_node

    def __select_path(self, node: Synset): # todo holly shit
        paths = node.hypernym_paths()
        for path in paths:
            if path[0].name() == 'entity.n.01':
                return path

    def __get_negative_sample(self, parent, node):
        random_node = random.choice(self.__all_synsets)
        i = 0
        while True:
            if random_node == parent:
                random_node = random.choice(self.__all_synsets)
            else:
                break
            i += 1
        if i > 0: print('Chose node in', i, 'iterations')
        return self.__select_path(random_node) + [node]

    def __collect_sample_paths(self, node: Synset):
        path = self.__select_path(node)
        res = [path]
        for i in range(self.__negative_samples_per_node):
            res.append(self.__get_negative_sample(path[-2], node))
        return res

    def __process_node(self, node, batches):
        samples = self.__collect_sample_paths(node)
        batches.append(samples)

    def prepare_ds(self, train_synsets: [Synset], batch_size=16):
        start = time.time()
        samples = []
        list(map(lambda n: self.__process_node(n, samples), train_synsets))
        for synset in train_synsets:
            self.__process_node(synset, samples)
        end = time.time()
        print('Finised creating samples in', end - start, 'seconds')

        pointer = 0
        batches = []
        while pointer < len(samples):
            positive_samples = []
            negative_samples = []
            i = 0
            while pointer < len(samples) and i < batch_size:
                positive_samples.append(samples[pointer][0])
                negative_samples += samples[pointer][1:]
                pointer += 1
                i += 1
            batches.append(PathsBatch(positive_samples, negative_samples))

        return batches
