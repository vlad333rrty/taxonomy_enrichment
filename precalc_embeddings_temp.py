from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


def read_terms(path):
    with open(path, 'r') as _file:
        wn_reader = WordNetDao.get_wn_30()
        res = []
        lines = _file.readlines()
        for line in lines:
            raw_term = line.strip()
            synsets = wn_reader.synsets(raw_term)
            for synset in synsets:
                res.append(Term(raw_term, synset.definition()))
        return res


def __get_temp_embeddings_provider(device='cpu'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    return TEMPEmbeddingProvider(tokenizer, bert_model, device)

terms = read_terms('data/datasets/diachronic-wordnets/en/no_labels_unprocessed')

