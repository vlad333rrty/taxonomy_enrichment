from abc import ABC, abstractmethod

import pg

class WordEmbeddingsDao:
    class Entry:
        def __init__(self, word, embedding):
            self.word = word
            self.embedding = embedding


        @classmethod
        def from_dict(cls, word2embedding: dict):
            entries = []
            for word in word2embedding:
                entries.append(cls(word, word2embedding[word]))
            return entries

    def __init__(self, db: pg.DB, table_name: str):
        self.__db = db
        self.table_name = table_name

    def find(self, word: str) -> [float]:
        res = self.__db.query_formatted('SELECT embedding FROM {} where word=%s'.format(self.table_name), [word]).dictresult()
        return res[0]['embedding']

    def find_embeddings_by_keys(self, words: [str]):
        where = '({})'.format(','.join(words))
        res = self.__db.query_formatted('SELECT * FROM {} where word in %s'.format(self.table_name), where).dictresult()
        dict_res = {}
        for r in res:
            dict_res[r['word']] = r['embedding']
        return dict_res

    def find_all_as_map(self):
        query_res = self.__db.query_formatted('SELECT * FROM {}'.format(self.table_name)).dictresult()
        res = {}
        for x in query_res:
            key = x['word']
            res[key] = x['embedding']
        return res

    def insert_many(self, entries: [Entry]):
        pattern = []
        values = []
        for entry in entries:
            pattern.append('(%s, %s)')
            values.append(entry.word)
            values.append(entry.embedding)
        self.__db.query_formatted('INSERT INTO {}(word, embedding) VALUES'.format(self.table_name) + ','.join(pattern), values)
