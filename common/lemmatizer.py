import pymorphy2


class Lemmatizer:
    def __init__(self):
        self.__analyzer = pymorphy2.MorphAnalyzer()

    def get_word_info(self, word):
        return self.__analyzer.parse(word)[0]

    def get_words_normal_form(self, word):
        return self.get_word_info(word).normal_form
