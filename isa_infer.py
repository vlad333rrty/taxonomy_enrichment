import pickle
import threading

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformerFactory
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddDataParser
from src.taxo_expantion_methods.is_a.IsAClassifier import IsAClassifier
from src.taxo_expantion_methods.is_a.graph_embedding_provider import IsAEmbeddingsProvider
from src.taxo_expantion_methods.parent_sum.embeddings_graph import EmbeddingsGraphNode, NodeEmbeddings
from src.taxo_expantion_methods.parent_sum.node_embeddings_provider import EmbeddingsGraphBERTNodeEmbeddingsProvider
from src.taxo_expantion_methods.utils.utils import paginate


class _TermSynsetAdapter:
    def __init__(self, term):
        self.__term = term

    def name(self):
        return '{}.n.00'.format(self.__term.value())

    def definition(self):
        return self.__term.definition()

    def value(self):
        return self.__term.value()

    def __str__(self):
        return f'TermSynsetAdapter({self.name()})'
    def __repr__(self):
        return f'TermSynsetAdapter({self.name()})'

device = 'cpu'
terms_path = 'data/datasets/semeval/training_data.csv'
load_path = 'data/models/isa/temp/isa_model_epoch_9'
result_path = 'data/results/semeval/isa/predicted.tsv'

terms = WordToAddDataParser.from_pandas(terms_path, '$')

res_terms = []
for term in terms:
    res_terms.append(_TermSynsetAdapter(Term(term.value, term.definition)))

model = IsAClassifier().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))

embeddings_graph = {}


def __format(candidates, token):
    names = list(map(lambda x: x.name(), candidates))
    return '{}\t{}\n'.format(token.value(), ','.join(names))


def write_to_file(str):
    with open(result_path, 'a') as file:
        file.write(str)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
embeddings_provider = IsAEmbeddingsProvider(embeddings_graph, bert_model, tokenizer, device)
wn = WordNetDao.get_wn_30()
all_synsets = list(wn.all_synsets('n'))
k = 5
with torch.no_grad():
    for term in res_terms:
        synset_and_term = list(
            map(
                lambda x: (x, term),
                all_synsets
            )
        )
        print('Got', len(synset_and_term), 'items for ranking')
        delta, embeds = performance.measure(lambda: embeddings_provider.get_embeddings(synset_and_term))
        print('Got embeddings in', delta, 'sec')
        scores = model(embeds)
        pair_with_score = list(zip(synset_and_term, scores.view(-1)))
        res = list(
            max(pair_with_score, key=lambda x:-x[1])
        )
        write_to_file(__format(res, term))
