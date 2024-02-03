from src.taxo_expantion_methods.dao.dao_factory import DaoFactory
from src.taxo_expantion_methods.dao.word_embeddings_dao import WordEmbeddingsDao
from gensim.models.fasttext import load_facebook_model

def run():
    model = load_facebook_model('/home/vlad333rrty/Downloads/cc.en.300.bin.gz')
    dao = DaoFactory.create_fasttext_subword_embeddings_enriched_dao()
    words_to_add = get_words_to_add()
    print('Found {} word to add'.format(len(words_to_add)))
    entries = []
    processed = 0
    for word in words_to_add:
        embedding = model[word]
        entries.append(WordEmbeddingsDao.Entry(word, embedding))
        processed += 1
        if processed % 100 == 0:
            print('{}% processed'.format(processed / len(words_to_add)))
    dao.insert_many(entries[:10_000])
    dao.insert_many(entries[10_000:])

def get_words_to_add():
    with open('data/wordnet_noun.terms', 'r') as file:
        lines = file.readlines()
    print('Got {} items'.format(len(lines)))
    ids_and_words = list(
        map(
            lambda x: (x[0], x[1].split('||')[0]),
            map(lambda x: x.split('\t'), lines)
        )
    )
    dao = DaoFactory.create_fasttext_subword_embeddings_dao()
    missed = set()
    processed = 0
    for id_and_word in ids_and_words:
        word = id_and_word[1]
        embedding = dao.find_or_none(word)
        if embedding is None:
            missed.add(word)
        processed += 1
        if processed % 100 == 0:
            print('{}% processed'.format(processed / len(lines) * 100))
    return missed


