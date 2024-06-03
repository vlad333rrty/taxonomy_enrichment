import random
import unittest

import numpy as np

from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.datasets_processing.FasttextVectorizer import FasttextVectorizer


class FasttextVectorizerTest(unittest.TestCase):
    class __FasttextModelMock:
        def __init__(self, dim):
            self.random = random.Random()
            self.dim = dim
            self.wv = self


        def __getitem__(self, item):
            return [1 for _ in range(self.dim)]


    def test_vectorizer(self):
        vectorizer = FasttextVectorizer(self.__FasttextModelMock(300))
        terms = [Term('term_{}'.format(i), 'definition {}'.format(i)) for i in range(10)]
        result = vectorizer.vectorize_terms(terms)
        for r in result.term_and_embed:
            self.assertListEqual(list(r[1]), [2 for _ in range(result.dimension)])