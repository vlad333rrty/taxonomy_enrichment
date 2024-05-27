import pickle
import random

import torch
from nltk.corpus import WordNetCorpusReader
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.path_selector import RuWnPathSelector, WnPathSelector
from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TEMP.temp_dataset_generator import TEMPDsCreator
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.TEMP.temp_loss import TEMPLoss
from src.taxo_expantion_methods.TEMP.trainer import TEMPTrainer
from src.taxo_expantion_methods.common.SynsetWrapper import RuSynsetWrapper
from src.taxo_expantion_methods.common.ru_wn_dao import RuWordnetDao
from src.taxo_expantion_methods.parent_sum.graph_embedding_provider import PrecalcEmbeddingsProvider
from src.taxo_expantion_methods.is_a.is_a_dataset_generator import IsADatasetGenerator
from src.taxo_expantion_methods.is_a.train import IsATrainer


def run_isa_model_training(embeddings_graph, device, epochs, res_path, model, wn_reader: WordNetCorpusReader, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    loss_fn = nn.BCELoss()
    all_synsets = list(wn_reader.all_synsets('n'))

    train_synsets, validation_synsets = train_test_split(all_synsets, train_size=0.8, test_size=0.05)

    ds_creator = IsADatasetGenerator(all_synsets)
    embedding_provider = PrecalcEmbeddingsProvider(embeddings_graph)
    trainer = IsATrainer(embedding_provider, res_path)
    trainer.train(
        model,
        optimizer,
        loss_fn,
        lambda: ds_creator.generate(train_synsets, batch_size),
        ds_creator.generate(validation_synsets, batch_size),
        epochs
    )

