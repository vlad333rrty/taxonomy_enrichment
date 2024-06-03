import random

from nltk.corpus import WordNetCorpusReader
from nltk.corpus.reader import Synset

from src.taxo_expantion_methods.common.Term import Term


class WordnetParser:
    @staticmethod
    def travers_synsets(all_synsets):
        terms, relations = WordnetParser.__create_relations_from_nodes(all_synsets)
        print(len(terms), len(relations))
        return terms, relations

    @staticmethod
    def __create_relations_from_nodes(synsets: [Synset]):
        relations = set()
        terms = set()
        for synset in synsets:
            hyponyms = synset.hyponyms()
            term = Term(synset.name(), synset.definition())
            terms.add(term)
            for hyponym in hyponyms:
                relations.add((synset.name(), hyponym.name()))
        return terms, relations

