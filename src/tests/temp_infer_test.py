import unittest

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformer
from src.taxo_expantion_methods.TEMP.synsets_provider import create_synsets_batch
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class TEMPTermInferencePerformerTest(unittest.TestCase):
    class _SynsetBatch:
        def __init__(self, synsets, hypernym_paths):
            self.__synsets = synsets
            self.__hypernym_paths = hypernym_paths

        def hypernym_paths(self):
            return self.__hypernym_paths

    class _Adapter:
        def __init__(self, value, definition, id_test):
            self.value = value
            self.definition = definition
            self.id_test = id_test

        def value(self):
            return self.value

        def definition(self):
            return self.definition

    class _EmbeddingProvider:
        def __init__(self, path2embed):
            self.__path2embed = path2embed

        def get_path_embeddings(self, paths):
            embeddings = []
            for path in paths:
                a, b = path[0].id_test, path[1].id_test
                embeddings.append(self.__path2embed[(a, b)])
            return embeddings

    def test(self):
        synset1 = self._Adapter('entity', 'entity def', 1)
        synset2 = self._Adapter('dog', 'dog def', 2)
        synset3 = self._Adapter('car', 'car def', 3)
        synset4 = self._Adapter('cat', 'cat def', 4)
        synsets_bathces = [
            self._SynsetBatch([synset1, synset2], [[synset1, synset1], [synset1, synset2]]),
            self._SynsetBatch([synset3, synset4], [[synset1, synset3], [synset1, synset4]])
        ]
        ep = self._EmbeddingProvider({
            (1, 1): 0.9,
            (1, 2): 0.5,
            (1, 3): 0.75,
            (1, 4): 0.5
        })
        model = lambda x: x
        inferer = TEMPTermInferencePerformer(synsets_bathces, ep)
        result = inferer.infer(model, [Term('test', 'test'), Term('test1', 'test1')])
        term1_res = result[0]
        term2_res = result[1]

        expected_path = [synset1, synset1]
        self.assertEquals(0.9, term1_res[0])
        self.assertEquals(expected_path, term1_res[1])

        self.assertEquals(0.9, term2_res[0])
        self.assertEquals(expected_path, term2_res[1])

