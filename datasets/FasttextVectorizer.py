class FasttextVectorizer:
    def __init__(self, model):
        self.__model = model

    @staticmethod
    def __get_simple_name(term):
        return term.split('.')[0]

    def vectorize_terms(self, terms):
        term_and_embed = []
        for term in terms:
            term_and_embed.append((term, self.__model.wv[FasttextVectorizer.__get_simple_name(term)]))
        return term_and_embed