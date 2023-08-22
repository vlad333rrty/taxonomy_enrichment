def extract_embedding(query_word: str, data_path: str):
    etalon = query_word.lower()
    with open(data_path) as file:
        for line in file:
            x = line.split(' ')
            sample = x[0].lower()
            if sample == etalon:
                return list(map(float, x[1:]))


def extract_embeddings(query_words: [str], data_path: str):
    words_set = set(map(lambda s: s.lower(), query_words))
    word_to_embedding = {}
    with open(data_path) as file:
        for line in file:
            x = line.split(' ')
            sample = x[0].lower()
            if sample in words_set:
                word_to_embedding[sample] = list(map(float, x[1:]))
    return word_to_embedding


def get_sentence_embedding(sentences: [[str]], model_name_or_path: str):
    from sentence_transformers import SentenceTransformer
    sentence_transformer = SentenceTransformer(model_name_or_path)
    return sentence_transformer.encode(sentences)