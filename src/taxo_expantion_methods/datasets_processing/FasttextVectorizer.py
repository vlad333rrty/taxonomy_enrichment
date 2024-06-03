from abc import ABC

import nltk
import numpy as np
from tqdm import tqdm


class VectorizationResult:
    def __init__(self, term_and_embed, dimension):
        self.term_and_embed = term_and_embed
        self.dimension = dimension

class TermVectorizer(ABC): # todo do we really need that???
    def vectorize_terms(self, terms):
        raise NotImplementedError()


class FasttextVectorizer(TermVectorizer):
    def __init__(self, model):
        self.__model = model
        self.__not_found_count = 0

    @staticmethod
    def __get_simple_name(term):
        return term.split('.')[0]

    def __get_embedding_for_token(self, token):
        embedding = self.__model.wv[token]
        if embedding is None:
            print('Failed to found embedding for token', token)
        return np.array(embedding)

    def __get_average_embedding(self, tokens):
        embeddings = list(
            filter(
                lambda x: x is not None,
                map(
                    self.__get_embedding_for_token,
                    tokens
                )
            )
        )
        return sum(embeddings) / len(embeddings)

    def vectorize_terms(self, terms):
        term_and_embed = []
        for term in terms:
            name_tokens = nltk.word_tokenize(term.value())
            def_tokens = nltk.word_tokenize(term.definition())
            name_embedding = self.__get_average_embedding(name_tokens)
            definition_embedding = self.__get_average_embedding(def_tokens)
            synset_embedding = name_embedding + definition_embedding
            term_and_embed.append((term.value(), synset_embedding))
        return VectorizationResult(term_and_embed, 300)

class BertVectorizer(TermVectorizer):
    def __init__(self, bert_embeddings_provider, wn_reader):
        self.__bert_embeddings_provider = bert_embeddings_provider
        self.__wn_reader = wn_reader

    def vectorize_terms(self, terms):
        term_and_embed = []
        for term in tqdm(terms):
            synset = self.__wn_reader.synset(term)
            term_and_embed.append((term, self.__bert_embeddings_provider.get_embedding(synset)))
        return VectorizationResult(term_and_embed, 768)
