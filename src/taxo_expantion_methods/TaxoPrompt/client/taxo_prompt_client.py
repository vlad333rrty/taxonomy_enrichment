import time

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForMaskedLM

from src.taxo_expantion_methods.TaxoPrompt.random_walk import create_extended_taxo_graph
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_dataset_creator import TaxoPromptDsCreator
from src.taxo_expantion_methods.TaxoPrompt.trainer import TaxoPromptTrainer
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


def run(device, epochs, load_path=None):
    wn_reader = WordNetDao.get_wn_20()
    root_synset = wn_reader.synset('entity.n.01')
    taxo_graph = create_extended_taxo_graph(root_synset)
    train_nodes = list(
        filter(lambda x: x.get_synset() != root_synset, taxo_graph.values())
    )
    X_train, X_test_val = train_test_split(train_nodes, train_size=0.6, test_size=0.3)
    X_test, X_val = train_test_split(X_test_val, train_size=0.5, test_size=0.5)

    print('Train size:', len(X_train))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    start = time.time()
    ds = TaxoPromptDsCreator(tokenizer).prepare_ds(X_train, 6, 5, 6)
    end = time.time()
    print('Created ds in', end - start)

    path = 'bert-base-uncased' if load_path is None else load_path
    bert_model = BertForMaskedLM.from_pretrained(path).to(device)

    optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-5)

    trainer = TaxoPromptTrainer(tokenizer, bert_model, optimizer, 'data/models/TaxoPrompt/checkpoints')
    trainer.train(ds, device, epochs)