import pickle

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig, BertModel

from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_loss import MLMLoss
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_model import TaxoPrompt
from src.taxo_expantion_methods.TaxoPrompt.trainer import TaxoPromptTrainer


def run(device, epochs, train_ration, ds_path, load_path=None):
    with open(ds_path, 'rb')as file:
        ds = pickle.load(file)
    X_train, X_test_val = train_test_split(ds, train_size=train_ration, test_size=0.3)

    print('Train size:', len(X_train))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    config = BertConfig()
    model = TaxoPrompt(config)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss = MLMLoss(config)

    trainer = TaxoPromptTrainer(
        tokenizer,
        bert_model,
        model,
        loss,
        optimizer,
        'data/models/TaxoPrompt/checkpoints')
    trainer.train(X_train, device, epochs)
