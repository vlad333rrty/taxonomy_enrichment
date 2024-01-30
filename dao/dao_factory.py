import pg

from dao.PermissionType import PermissionType
from dao.word_embeddings_dao import WordEmbeddingsDao


class DaoFactory:
    __DB = pg.DB(dbname='postgres', host='localhost', user='Vlad', port=32768, passwd='123456')

    @staticmethod
    def create_word_embeddings_dao():
        return WordEmbeddingsDao(DaoFactory.__DB, 'word_embeddings', PermissionType.READ_ONLY)

    @staticmethod
    def create_poincare_word_embeddings_dao():
        return WordEmbeddingsDao(DaoFactory.__DB, 'poincare_word_embeddings')

    @staticmethod
    def create_poincare_word_embeddings_dao_2():
        return WordEmbeddingsDao(DaoFactory.__DB, 'poincare_word_embeddings_2')

    @staticmethod
    def create_node_embeddings_dao(permission_type=PermissionType.READ_WRITE):
        return WordEmbeddingsDao(DaoFactory.__DB, 'node_embeddings', permission_type)

    @staticmethod
    def create_node_embeddings_dao_2():
        return WordEmbeddingsDao(DaoFactory.__DB, 'node_embeddings_2')

    @staticmethod
    def create_fasttext_subword_embeddings_dao():
        return WordEmbeddingsDao(DaoFactory.__DB, 'fasttext_embeddings_subword')

    @staticmethod
    def create_fasttext_subword_embeddings_enriched_dao():
        return WordEmbeddingsDao(DaoFactory.__DB, 'fasttext_embeddings_subword_enriched')

