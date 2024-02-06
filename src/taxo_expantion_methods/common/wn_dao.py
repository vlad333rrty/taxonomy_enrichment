from nltk.corpus import WordNetCorpusReader


class WordNetDao:
    @staticmethod
    def get_wn_20():
        return WordNetCorpusReader('data/wordnets/WordNet-2.0/dict', None)

    @staticmethod
    def get_wn_30():
        return WordNetCorpusReader('data/wordnets/WordNet-3.0/dict', None)
