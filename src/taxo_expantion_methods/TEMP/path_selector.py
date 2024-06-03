import random

from nltk.corpus.reader import Synset


# FIXME this approach is bad, no path selectors should exist

class SubgraphPathSelector:
    def __init__(self, root: Synset):
        self.__root = root


    def select_path(self, node):
        paths = node.hypernym_paths()
        selected_path = random.choice(list(filter(lambda x: self.__root in x, paths)))
        i = 0
        while selected_path[i] != self.__root:
            i += 1
        return selected_path[i:]

class DummyPathSelector:
    def select_path(self, node):
        return node.hypernym_paths()

class WnPathSelector:
    def select_path(self, node):
        paths = node.hypernym_paths()
        return random.choice(list(
            filter(
                lambda x: x[0].name() == 'entity.n.01',
                paths
            ))
        )


class RuWnPathSelector:
    def __init__(self, root_id: str):
        self.__root_id = root_id


    def select_path(self, node):
        paths = node.hypernym_paths()
        return random.choice(list(
            filter(
                lambda x: x[0].id() == self.__root_id,
                paths
            ))
        )