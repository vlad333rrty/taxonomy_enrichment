import random
import time
from multiprocessing.pool import ThreadPool

from nltk.corpus.reader import Synset

from src.taxo_expantion_methods.TEMP.plot_monitor import PlotMonitor


class PathsBatch:
    def __init__(self, positive_paths, negative_paths):
        self.positive_paths = positive_paths
        self.negative_paths = negative_paths


class TEMPDsCreator:
    def __init__(self, all_synsets, negative_samples_per_node=4):
        self.__all_synsets = all_synsets
        self.__negative_samples_per_node = negative_samples_per_node

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
        return random_node.hypernym_paths()[0] + [node]

    def __collect_sample_paths(self, node: Synset):
        path = node.hypernym_paths()[0]
        res = [path]
        for i in range(self.__negative_samples_per_node):
            res.append(self.__get_negative_sample(path[-2], node))
        return res

    def __process_node(self, node, batches):
        print('Processing node', node)
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
