import nltk

from engines.word_to_add_data import WordToAddData

__RAW_POS_TO_UNIVERSAL_POS = {
    'noun': 'NN',
    'verb': 'VBZ'
}

__SELECTED_POS = {
    'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR	', 'JJS'
}


def __get_nltk_pos(raw_pos: str):
    return __RAW_POS_TO_UNIVERSAL_POS[raw_pos]


def gather_essential_words(words_to_add: [WordToAddData], limit: int):
    essential_words_per_word = {}
    num2word = {}
    for data in words_to_add:
        pos = data.pos
        definition = data.definition
        word = data.value
        id = data.num
        essential_words = __get_essential_words(__get_nltk_pos(pos), nltk.word_tokenize(definition), limit)

        essential_words_per_word[id] = (essential_words, pos)
        num2word[id] = word
    return essential_words_per_word, num2word


def __get_essential_words(pos, definition, limit: int):
    definiton_words_infos = nltk.pos_tag(definition)
    return list(map(lambda x: x[0], filter(lambda x: x[1] in __SELECTED_POS, definiton_words_infos)))[:limit]
