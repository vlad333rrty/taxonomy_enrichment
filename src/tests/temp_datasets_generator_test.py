import unittest

from nltk.corpus import WordNetCorpusReader
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.datasets_processing.temp_dataset_generator import TEMPDsCreator


class TEMPSamplesCreatorTest(unittest.TestCase):
    def test(self):
        wn_reader = WordNetCorpusReader('data/wordnets/WordNet-2.0/dict', None)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        samples_creator = TEMPDsCreator(list(wn_reader.all_synsets('n')), tokenizer, bert_model)
        batches = samples_creator.prepare_ds([wn_reader.synset('dog.n.01'), wn_reader.synset('cat.n.01')])
        print(len(batches))

        loader = DataLoader(batches, batch_size=16)
        for x in loader:
            print(len(x))
