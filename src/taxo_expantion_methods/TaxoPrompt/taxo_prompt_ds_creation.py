import pickle

from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TaxoPrompt.random_walk import create_extended_taxo_graph
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_dataset_creator import TaxoPromptDsCreator
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


def __get_train_samples(synset_to_node):
    return list(
        map(
            lambda k: synset_to_node[k],
            filter(
                lambda key: key.name() != 'entity.n.01',
                synset_to_node
            )
        )
    )


def create_ds():
    wn = WordNetDao.get_wn_20()
    synset_to_node = create_extended_taxo_graph(wn.synset('entity.n.01'))
    train_nodes = __get_train_samples(synset_to_node)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    ds_creator = TaxoPromptDsCreator(tokenizer)
    return ds_creator.prepare_ds(train_nodes, 6, 5)


def dump(obj):
    with open('data/datasets/taxo-prompt/nouns/ds', 'wb') as file:
        pickle.dump(obj, file)


ds = create_ds()
dump(ds)
