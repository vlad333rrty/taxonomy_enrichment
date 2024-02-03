import random

from nltk.corpus import WordNetCorpusReader
from nltk.corpus.reader import Synset


class WordnetParser:
    def __init__(self, wordnet_reader: WordNetCorpusReader):
        self.__wn_reader = wordnet_reader

    def traverse_nouns(self):
        """
        Traverses wordnet nouns and gather their relations
        :return: pair such as (terms, relations)
        """
        all_synsets = self.__wn_reader.all_synsets('n')
        terms, relations = self.__create_relations_from_nodes(all_synsets)
        print(len(terms), len(relations))
        return terms, relations
        # return self.__traverse_wordnet('entity.n.01')

    def __travers_synset_based(self, pos):
        all_synsets = list(self.__wn_reader.all_synsets(pos))
        relations = self.__create_relations_from_nodes(all_synsets)
        terms = list(map(lambda x: x.name(), all_synsets))
        return terms, relations

    def traverse_verbs(self):
        """
        :return: all verbs in the wordnet and their relations
        """
        return self.__travers_synset_based('v')

    @staticmethod
    def __create_relations_from_nodes(synsets: [Synset]):
        relations = set()
        terms = set()
        for synset in synsets:
            hyponyms = synset.hyponyms()
            terms.add(synset.name())
            for hyponym in hyponyms:
                relations.add((synset.name(), hyponym.name()))
        return terms, relations

    @staticmethod
    def __create_relations_from_nodes_and_gather_terms(synsets: [Synset], max_relations=20_000): # todo govno
        relations = set()
        terms = set()
        for synset in synsets:
            hyponyms = synset.hyponyms()
            terms.add(synset.name())
            for hyponym in hyponyms:
                relations.add((synset.name(), hyponym.name()))
                terms.add(hyponym.name())
                if len(relations) == max_relations:
                    return terms, relations
        return terms, relations

    def __traverse_wordnet(self, root_term) -> ({str}, {(str, str)}):
        relations = set()
        root = self.__wn_reader.synset(root_term)
        terms = self.__append_pairs(root, relations)
        return terms, relations

    @staticmethod
    def __append_pairs(root, pairs):
        stack = [root]
        used = set()
        while len(stack) > 0:
            current = stack.pop()
            if current.name() in used:
                continue
            for w in current.hyponyms():
                pairs.add((current.name(), w.name()))
                stack.append(w)
            used.add(current.name())
        return used
