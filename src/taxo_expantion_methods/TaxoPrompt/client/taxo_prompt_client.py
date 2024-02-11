import pickle
import time

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig

from src.taxo_expantion_methods.TaxoPrompt.random_walk import create_extended_taxo_graph
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_dataset_creator import TaxoPromptDsCreator
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_loss import MLMLoss
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_model import TaxoPrompt
from src.taxo_expantion_methods.TaxoPrompt.trainer import TaxoPromptTrainer
from src.taxo_expantion_methods.common.wn_dao import WordNetDao

def run(device, epochs, train_ration, ds_path, load_path=None):
    with open(ds_path, 'rb')as file:
        ds = pickle.load(file)
    X_train, X_test_val = train_test_split(ds, train_size=train_ration, test_size=0.3)

    print('Train size:', len(X_train))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    path = 'bert-base-uncased' if load_path is None else load_path
    bert_model = BertForMaskedLM.from_pretrained(path).to(device)

    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=1e-5)
    config = BertConfig()
    model = TaxoPrompt(config)
    loss = MLMLoss(config)

    trainer = TaxoPromptTrainer(
        tokenizer,
        bert_model,
        model,
        loss,
        optimizer,
        'data/models/TaxoPrompt/checkpoints')
    trainer.train(X_train, device, epochs)
