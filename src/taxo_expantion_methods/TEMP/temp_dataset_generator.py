import random
import time

from nltk.corpus.reader import Synset


class PathsBatch:
    def __init__(self, positive_paths, negative_paths):
        self.positive_paths = positive_paths
        self.negative_paths = negative_paths


class TEMPDsCreator:
    def __init__(self, all_synsets, path_selector, negative_samples_per_node=1):
        self.__all_synsets = all_synsets
        self.__path_selector = path_selector
        self.__negative_samples_per_node = negative_samples_per_node

    def __get_negative_sample(self, parents, node):
        random_node = random.choice(self.__all_synsets)
        i = 0
        while True:
            if random_node in parents:
                random_node = random.choice(self.__all_synsets)
            else:
                break
            i += 1
        if i > 0: print('Chose node in', i, 'iterations')
        return self.__path_selector.select_path(random_node) + [node]

    def __collect_sample_paths(self, node: Synset):
        path = self.__path_selector.select_path(node)
        res = [path]
        for i in range(self.__negative_samples_per_node):
            res.append(self.__get_negative_sample(set(node.hypernyms()), node))
        return res

    def __process_node(self, node, batches):
        samples = self.__collect_sample_paths(node)
        batches.append(samples)

    def prepare_ds(self, train_synsets: [Synset], batch_size):
        start = time.time()
        samples = []
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
