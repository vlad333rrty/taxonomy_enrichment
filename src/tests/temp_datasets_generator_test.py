import unittest

from nltk.corpus import WordNetCorpusReader
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.path_selector import WnPathSelector
from src.taxo_expantion_methods.TEMP.temp_dataset_generator import TEMPDsCreator


class TEMPSamplesCreatorTest(unittest.TestCase):
    def test(self): # todo fix
        wn_reader = WordNetCorpusReader('data/wordnets/WordNet-2.0/dict', None)
        samples_creator = TEMPDsCreator(list(wn_reader.all_synsets('n')), WnPathSelector())
        batch_size = 16
        batches = samples_creator.prepare_ds([wn_reader.synset('dog.n.01'), wn_reader.synset('cat.n.01')], batch_size)
        for batch in batches:
            self.assertEquals(batch_size, len(batch.positive_paths))
            self.assertEquals(batch_size, len(batch.negative_paths))
