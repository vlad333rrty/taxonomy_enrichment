import re

import pg

from src.taxo_expantion_methods.dao.PermissionType import PermissionType


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

    def __init__(self, db: pg.DB, table_name: str, permission_type = PermissionType.READ_WRITE):
        self.__db = db
        self.table_name = table_name
        self.permission_type = permission_type

    def contains(self, word: str) -> bool:
        res = self.__db.query_formatted('SELECT * FROM {} where word=%s'.format(self.table_name), [word]).dictresult()
        return len(res) > 0

    def find_or_none(self, word: str) -> [float]:
        res = self.__db.query_formatted('SELECT embedding FROM {} where word=%s'.format(self.table_name), [word]).dictresult()
        if len(res) == 0:
            return None
        return res[0]['embedding']

    def find_by_keys_as_map(self, words: [str]):
        where = ','.join(
            map(
                lambda w: "'{}'".format(w),
                map(
                    lambda w: re.sub('\'', '\'\'', w),
                    words
                )
            )
        )
        res = self.__db.query('SELECT * FROM {} where word in ({})'.format(self.table_name, where)).dictresult()
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
        if self.permission_type != PermissionType.READ_WRITE:
            raise PermissionError('Cannot write into table. Permission denied: ', self.permission_type)
        pattern = []
        values = []
        for entry in entries:
            pattern.append('(%s, %s)')
            values.append(entry.word)
            values.append(entry.embedding)
        self.__db.query_formatted('INSERT INTO {}(word, embedding) VALUES'.format(self.table_name) + ','.join(pattern), values)
