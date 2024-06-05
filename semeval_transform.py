import pickle
import threading

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from precalc_embeddings_temp import GraphLoader
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
result_path = 'data/isa_res.tsv'

terms = WordToAddDataParser.from_pandas(terms_path, '$')

res_terms = []
for term in terms:
    res_terms.append(_TermSynsetAdapter(Term(term.value, term.definition)))

model = IsAClassifier().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
model.eval()

embeddings_graph = GraphLoader.load_graph('data/embeddings_graph_no_sum.pkl')

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

def get_score(synset, term):
    emb = embeddings_provider.get_embeddings([(synset, term)])
    return model(emb)

def traverse(root, term, k=1):
    stack = [root]
    path = []
    used = set()
    while len(stack) > 0:
        u = stack.pop()
        children_and_score = []
        if len(u.hyponyms()) == 0:
            continue
        for child in u.hyponyms():
            if child not in used:
                children_and_score.append((child, get_score(child, term)))
        best_candidate = max(children_and_score, key=lambda x:x[1])
        stack.append(best_candidate[0])
        path.append(best_candidate)
        used.add(u)
    return path

root = wn.synset('entity.n.01')
for term in res_terms:
    with torch.no_grad():
        path = traverse(root, term)
    candidates = list(
        map(
            lambda x:x[0],
            path
        )
    )
    write_to_file(__format(candidates, term))



