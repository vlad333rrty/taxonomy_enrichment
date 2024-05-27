import random
import time

from nltk.corpus.reader import Synset

from src.taxo_expantion_methods.utils.utils import paginate


class IsABatch:
    def __init__(self, positive_samples, negative_samples):
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples


class IsADatasetGenerator:

    def __init__(self, all_synsets):
        self.__all_synsets = all_synsets

    def __get_random_elems(self, elems, k): #dangerous
        if len(elems) < k:
            return set(elems)
        unique = set()
        while len(unique) < k:
            unique.add(random.choice(elems))
        return list(unique)

    def __get_samples_for_node(self, node: Synset, mix_ratio=0.1, samples_count=4):
        paths = node.hypernym_paths()
        all_positive_nodes = set(
            [x for path in paths for x in path]
        )
        if len(all_positive_nodes) < samples_count:
            return [], []
        chosen = random.choice(paths)
        negative = set()
        for i in range(len(chosen) - 1):
            candidate = chosen[i]
            children = candidate.hyponyms()
            negative_candidates = list(
                filter(
                    lambda x: x not in all_positive_nodes,
                    children
                )
            )
            negative.update(negative_candidates)
        mix_count = int(len(negative) * mix_ratio)
        i = 0
        while i < mix_count or len(negative) < len(all_positive_nodes):
            random_node = random.choice(self.__all_synsets)
            while random_node in all_positive_nodes:
                random_node = random.choice(self.__all_synsets)
            negative.add(random_node)
            i += 1
        positive = list(all_positive_nodes)
        return self.__get_random_elems(positive, samples_count), self.__get_random_elems(list(negative), samples_count)

    def generate(self, train_synsets, batch_size):
        start = time.time()
        batch = []
        for synset in train_synsets:
            pos, neg = self.__get_samples_for_node(synset)
            batch += list(map(lambda p: (p, synset), pos)) + list(map(lambda p: (p, synset), neg)) # todo ...
        batches = paginate(batch, batch_size)
        end = time.time()
        print('Got {} batches in {}sec'.format(len(batches), end - start))
        return batches
