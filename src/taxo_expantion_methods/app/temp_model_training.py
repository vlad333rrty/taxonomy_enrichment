import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TEMP.temp_dataset_generator import TEMPDsCreator
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.TEMP.temp_loss import TEMPLoss
from src.taxo_expantion_methods.TEMP.trainer import TEMPTrainer
from src.taxo_expantion_methods.common.SynsetWrapper import RuSynsetWrapper
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


def run_temp_model_training(device, epochs, res_path, model, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    loss_fn = TEMPLoss(0.2).to(device)
    wn_reader = WordNetDao.get_wn_20()
    # all_synsets = list(wn_reader.all_synsets(wn_reader.NOUN))
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))

    train_synsets, test_synsets = train_test_split(all_synsets, train_size=0.75, test_size=0.1)
    print('Train/test:', len(train_synsets), len(test_synsets))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    ds_creator = TEMPDsCreator(all_synsets)
    embedding_provider = TEMPEmbeddingProvider(tokenizer, bert_model, device)
    trainer = TEMPTrainer(embedding_provider, res_path)
    trainer.train(model, optimizer, loss_fn, lambda: ds_creator.prepare_ds(train_synsets, batch_size),
                  ds_creator.prepare_ds(test_synsets, batch_size), epochs)


def run_temp_model_training_ru(device, epochs, res_path, model, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    loss_fn = TEMPLoss(0.2).to(device)
    dao = WordNetDao.get_ru_wn_20()
    all_synsets = list(map(RuSynsetWrapper, dao.synsets))

    train_synsets, test_synsets = train_test_split(all_synsets, train_size=0.8, test_size=0.2)
    print('Train/test:', len(train_synsets), len(test_synsets))

    tokenizer = BertTokenizer.from_pretrained('ai-forever/sbert_large_nlu_ru')
    bert_model = BertModel.from_pretrained('ai-forever/sbert_large_nlu_ru').to(device)
    ds_creator = TEMPDsCreator(all_synsets)
    embedding_provider = TEMPEmbeddingProvider(tokenizer, bert_model, device)
    trainer = TEMPTrainer(embedding_provider, res_path)
    trainer.train(model, optimizer, loss_fn, lambda: ds_creator.prepare_ds(train_synsets, batch_size),
                  ds_creator.prepare_ds(test_synsets, batch_size), epochs)