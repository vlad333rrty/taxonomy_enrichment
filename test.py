import pickle

import torch
from gensim.models.fasttext import load_facebook_model
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from typing import Dict

from precalc_embeddings_temp import GraphLoader
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddDataParser
from src.taxo_expantion_methods.parent_sum.embeddings_graph import EmbeddingsGraphNode, NodeEmbeddings
from src.taxo_expantion_methods.parent_sum.node_embeddings_provider import EmbeddingsGraphBERTNodeEmbeddingsProvider, \
    EmbeddingsGraphFasttextNodeEmbeddingsProvider
from src.taxo_expantion_methods.utils.similarity import cos_sim

wn = WordNetDao.get_wn_30()
device = 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
res_path = 'data/predicted_fasttext.tsv'

# cos_sim = torch.nn.CosineSimilarity().to(device)

def __get_terms(terms_path):
    return list(
        map(
            lambda term: Term(term.value, term.definition, term.pos),
            WordToAddDataParser.from_pandas(terms_path, '$')
        )
    )

def __update_k_best(buffer, score, path):
    min = 1 << 32
    index = -1
    for i in range(len(buffer)):
        x = buffer[i][0]
        if x < score and min > x:
            min = x
            index = i
    if index > -1:
        buffer[index] = (score, path)

def __recover_path(id2node: Dict[str,EmbeddingsGraphNode], node:EmbeddingsGraphNode, node_embeddings_holder: NodeEmbeddings):
    path = [node.get_synset_name()]
    pointer = node_embeddings_holder.parent
    while pointer is not None:
        path.append(pointer)
        pointer = id2node[pointer].get_synset_name()

    return list(reversed(path))

def __get_best_nodes_score(node:EmbeddingsGraphNode, candidate_embedding):
    node_embeddings = node.get_embeddings()
    max_score = -1
    for embedding in node_embeddings:
        score = cos_sim(embedding.embedding, candidate_embedding)
        if max_score < score:
            max_score = score

    return max_score


def __get_candidates(term: Term, id2node, term_embedding, k=5):
    all_synsets = wn.all_synsets('n' if term.part_of_speech() == 'noun' else 'v')
    results = []
    not_found = []
    with torch.no_grad():
        for synset in tqdm(all_synsets):
            node = id2node.get(synset.name())
            if node is None:
                not_found.append(synset)
                continue
            score = __get_best_nodes_score(node, term_embedding)
            results.append((score, synset.name()))
    print('Failed to find {} synsets'.format(len(not_found)))
    return sorted(results, key=lambda x: -x[0])[:k]


def __write_result(path, term, anchors):
    res_str = '{}\t{}\n'.format(term, ','.join(anchors))
    with open(path, 'a') as file:
        file.write(res_str)


class _TermSynsetAdapter:
    def __init__(self, term):
        self.__term = term

    def name(self):
        return '{}.n.00'.format(self.__term.value())

    def definition(self):
        return self.__term.definition()

    def __str__(self):
        return f'TermSynsetAdapter({self.name()})'

def run(res_path):
    terms = __get_terms('data/datasets/semeval/training_data.csv')
    anchors = []
    graph_path = 'data/embeddings_graph_fasttext.pkl'
    with open(graph_path, 'rb') as file: graph = pickle.load(file)
    # embedding_provider = EmbeddingsGraphBERTNodeEmbeddingsProvider(bert_model, tokenizer, device)
    delta, model = performance.measure(
        lambda: load_facebook_model('/home/vlad333rrty/Downloads/cc.en.300.bin.gz'))
    print('Fasttext model loaded in', delta, 'seconds')
    embedding_provider = EmbeddingsGraphFasttextNodeEmbeddingsProvider(model)
    for term in terms:
        embedding = embedding_provider.get_embedding(_TermSynsetAdapter(term))
        delta, candidates = performance.measure(lambda: __get_candidates(term, graph, embedding))
        print('Processed term in {}s'.format(delta))
        print(candidates)
        current_anchors = list(map(lambda x: x[1], candidates))
        __write_result(res_path, term, current_anchors)


run(res_path)
