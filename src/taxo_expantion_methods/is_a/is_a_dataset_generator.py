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

    def __get_samples_for_node(self, node: Synset, mix_ratio=0.1, samples_count=8):
        paths = node.hypernym_paths()
        all_positive_nodes = set(
            [x for path in paths for x in path]
        )
        if len(all_positive_nodes) < samples_count:
            return [], []
        chosen = random.choice(paths)
        negative = []
        for i in range(len(chosen) - 1):
            candidate = chosen[i]
            children = candidate.hyponyms()
            negative_candidates = list(
                filter(
                    lambda x: x not in all_positive_nodes,
                    children
                )
            )
            negative += negative_candidates
        mix_count = int(len(negative) * mix_ratio)
        for i in range(mix_count):
            random_node = random.choice(self.__all_synsets)
            while random_node in all_positive_nodes:
                random_node = random.choice(self.__all_synsets)
            negative.append(random_node)
        positive = list(all_positive_nodes)
        random.shuffle(positive)
        random.shuffle(negative)
        return positive[:samples_count], negative[:samples_count]

    def generate(self, train_synsets, batch_size=32):
        start = time.time()
        pos_batch, neg_batch = [], []
        for synset in train_synsets:
            pos, neg = self.__get_samples_for_node(synset)
            assert len(pos) == len(neg)
            pos_batch += pos
            neg_batch += neg
        pos_batches = paginate(pos_batch, batch_size)
        neg_batches = paginate(neg_batch, batch_size)
        batches = []
        for i in range(len(pos_batches)):
            batches.append(IsABatch(pos_batches[i], neg_batches[i]))
        end = time.time()
        print('Got {} batches in {}sec'.format(len(batches), end - start))
        return batches
