import random
import re
from collections import deque

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class RandomTaxonomySynset:
    def __init__(self, name, definition, parent, hyponyms):
        self.__name = name
        self.__definition = definition
        self.__parent = parent
        self.__hyponyms = hyponyms

    def name(self):
        return self.__name

    def definition(self):
        return self.__definition

    def hypernym_paths(self):
        path = []
        cursor = self.__parent
        while cursor is not None:
            path.append(cursor)
            cursor = cursor.parent()
        return [path]

    def parent(self):
        return self.__parent

    def hypernyms(self):
        return [self.__parent]

    def hyponyms(self):
        return self.__hyponyms

    def __repr__(self):
        return f'RandomTaxonomySynset(name={self.__name},definition={self.__definition},hyponyms_count={len(self.__hyponyms)})'


class RandomTaxonomySynsetFactory:
    def __init__(self, wn):
        self.__wn = wn

    def create(self, term: str, parent) -> RandomTaxonomySynset:
        name = '{}.n.01'.format(term)
        synsets = self.__wn.synsets(term)
        if len(synsets) == 0:
            print(term)
        definition = random.choice(synsets).definition()
        return RandomTaxonomySynset(name, definition, parent, [])


def generate_random_taxonomy(synset_factory: RandomTaxonomySynsetFactory, nodes):
    iterator = iter(nodes)
    fst = next(iterator)
    queue = deque()
    root = synset_factory.create(fst, None)
    queue.append(root)
    while len(queue) > 0:
        parent = queue.popleft()
        hyponyms_count = random.randint(2, 4)
        for _ in range(hyponyms_count):
            child = next(iterator, None)
            if child is None:
                break
            synset = synset_factory.create(child, parent)
            queue.append(synset)
            parent.hyponyms().append(synset)
    return root

def read_terms(path):
    terms = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split('\t')
            term = data[1]
            terms.append(re.sub(' ', '_', term))
    return terms

nodes = read_terms('data/datasets/semeval-task13/food_wordnet_en.terms')
wn = WordNetDao.get_wn_30()
root = wn.synset('food.n.01')
food_taxonomy = SynsetsProvider.get_all_synsets_with_common_root(root)
print(len(food_taxonomy))