from abc import ABC

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

    @staticmethod
    def __get_simple_name(term):
        return term.split('.')[0]

    def vectorize_terms(self, terms):
        term_and_embed = []
        for term in terms:
            term_and_embed.append((term, self.__model.wv[FasttextVectorizer.__get_simple_name(term)]))
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
