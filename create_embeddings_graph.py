import pickle

import torch
from gensim.models.fasttext import load_facebook_model
from transformers import BertModel, BertTokenizer

from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.parent_sum.embeddings_graph import EmbeddingsGraphProvider
from src.taxo_expantion_methods.parent_sum.node_embeddings_provider import EmbeddingsGraphBERTNodeEmbeddingsProvider, \
    EmbeddingsGraphFasttextNodeEmbeddingsProvider

# device = 'cpu'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
wn = WordNetDao.get_wn_30()
result_file = 'data/embeddings_graph_fasttext.pkl'

delta, model = performance.measure(
    lambda: load_facebook_model('/home/vlad333rrty/Downloads/cc.en.300.bin.gz'))
print('Fasttext model loaded in', delta, 'seconds')
embeddings_provider = EmbeddingsGraphFasttextNodeEmbeddingsProvider(model)
graph_provider = EmbeddingsGraphProvider(embeddings_provider, lambda parent_emb,node_emb: node_emb)

with torch.no_grad():
    delta, n2n = performance.measure(lambda: graph_provider.from_wordnet(wn.synset('entity.n.01')))
print('Finished in', delta)

with open(result_file, 'wb') as file:
    pickle.dump(n2n, file)

