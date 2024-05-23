import random

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.path_selector import RuWnPathSelector, WnPathSelector
from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TEMP.temp_dataset_generator import TEMPDsCreator
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.TEMP.temp_loss import TEMPLoss
from src.taxo_expantion_methods.TEMP.trainer import TEMPTrainer
from src.taxo_expantion_methods.common.SynsetWrapper import RuSynsetWrapper
from src.taxo_expantion_methods.common.ru_wn_dao import RuWordnetDao


def run_temp_model_training(device, epochs, res_path, model, wn_reader, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    loss_fn = TEMPLoss(0.5).to(device)
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))

    train_synsets, validation_synsets = train_test_split(all_synsets, train_size=0.75, test_size=0.05)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    ds_creator = TEMPDsCreator(all_synsets, WnPathSelector())
    embedding_provider = TEMPEmbeddingProvider(tokenizer, bert_model, device)
    trainer = TEMPTrainer(embedding_provider, res_path)
    trainer.train(
        model,
        optimizer,
        loss_fn,
        lambda: ds_creator.prepare_ds(train_synsets, batch_size),
        ds_creator.prepare_ds(validation_synsets, batch_size),
        epochs
    )


def run_temp_model_training_ru(device, epochs, res_path, model, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    loss_fn = TEMPLoss(0.2).to(device)
    dao = RuWordnetDao.get_ru_wn_20()
    all_synsets = list(
        filter(
            lambda x: len(x.definition()) > 0,
            map(RuSynsetWrapper, dao.synsets)
        )
    )

    train_synsets, test_synsets = train_test_split(all_synsets, train_size=0.8, test_size=0.2)
    print('Train/test:', len(train_synsets), len(test_synsets))

    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased').to(device)

    noun_roots = ['134850-N', '106646-N', '106613-N', '153782-N', '130515-N', '106508-N', '132964-N', '130112-N',
                  '107561-N', '100106-N', '3484-N', '106846-N', '106509-N', '153471-N']

    ds_creator = TEMPDsCreator(all_synsets, RuWnPathSelector(random.choice(noun_roots)))
    embedding_provider = TEMPEmbeddingProvider(tokenizer, bert_model, device)
    trainer = TEMPTrainer(embedding_provider, res_path)
    def train_synsets_provider(): return train_test_split(all_synsets, train_size=0.8, test_size=0.1)[0]
    trainer.train(
        model,
        optimizer,
        loss_fn,
        lambda: TEMPDsCreator(all_synsets, RuWnPathSelector(random.choice(noun_roots))).prepare_ds(train_synsets_provider(), batch_size),
        ds_creator.prepare_ds(test_synsets, batch_size), epochs)