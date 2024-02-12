import pickle

from transformers import BertTokenizer

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TaxoPrompt.random_walk import create_extended_taxo_graph
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_dataset_creator import TaxoPromptDsCreator
from src.taxo_expantion_methods.common.wn_dao import WordNetDao

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

ds_c = TaxoPromptDsCreator(tokenizer)
root = WordNetDao.get_wn_20().synset('entity.n.01')
taxo_graph = create_extended_taxo_graph(root)
train_nodes = list(
    filter(lambda x: x.get_synset() != root, taxo_graph.values())
)

s = ds_c.prepare_ds(train_nodes, 6, 5)
with open('data/datasets/taxo-prompt/nouns/ds2', 'wb') as file:
    pickle.dump(s, file)