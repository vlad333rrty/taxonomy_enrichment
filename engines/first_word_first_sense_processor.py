import nltk
from engines.essential_words_gathering_utils import gather_essential_words
from engines.word_to_add_data import WordToAddDataParser


def gather_first_synsets(path: str, sep: str):
    result = []
    data_for_print = []
    essential_words_per_word, word_to_num = gather_essential_words(WordToAddDataParser.from_pandas(path, sep), 10)
    wordnet = nltk.corpus.wordnet31
    operation = 'attach'  # todo temp solution
    for key in essential_words_per_word:
        essential_words = essential_words_per_word[key][0]
        if len(essential_words) > 0:
            synsets = []
            i = 0
            while len(synsets) == 0 and i < len(essential_words):
                synsets = wordnet.synsets(essential_words[i])
                i += 1
            if len(synsets) > 0:
                result.append([
                    word_to_num[key],
                    synsets[0],
                    operation
                ])
            data_for_print.append([
                key,
                essential_words[0],
                synsets
            ])

    return result, data_for_print
