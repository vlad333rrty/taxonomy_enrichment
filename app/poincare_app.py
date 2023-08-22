import pg

from dao.word_embeddings_dao import WordEmbeddingsDao
from engines.poincare_embeddings_based_processor import PoincareEmbeddingsBasedProcessor
from engines.word_to_add_data import WordToAddDataParser


def run():
    db = pg.DB(dbname='postgres', host='localhost', user='Vlad', port=32768,
          passwd='123456')
    dao = WordEmbeddingsDao(db, 'word_embeddings')
    processor = PoincareEmbeddingsBasedProcessor(dao)
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    processor.process(words_to_add_data)


run()