import torch
from nltk.corpus import WordNetCorpusReader
from nltk.corpus.reader import Synset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.TEMP.temp_loss import TEMPLoss
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.TEMP.trainer import TEMPTrainer
from src.taxo_expantion_methods.common.configuration import Configuration
from src.taxo_expantion_methods.datasets_processing.temp_dataset_generator import TEMPDsCreator


def dfs(root: Synset):
    stack = [root]
    used = set()
    visited = []
    while len(stack) > 0:
        u = stack.pop()
        visited.append(u)
        for child in u.hyponyms():
            if child not in used:
                stack.append(child)
        used.add(u)
    return visited


def run():
    model = TEMP(768, 192)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = TEMPLoss(0.5)
    wn_reader = WordNetCorpusReader(Configuration.WORDNET_20_PATH, None)
    all_synsets = list(wn_reader.all_synsets(wn_reader.NOUN))
    # allowed_synsets = dfs(wn_reader.synset('entity.n.01'))

    train_synsets, test_synsets = train_test_split(all_synsets, train_size=0.6, test_size=0.02)
    print('Train/test:', len(train_synsets), len(test_synsets))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    ds_creator = TEMPDsCreator(all_synsets)
    embedding_provider = TEMPEmbeddingProvider(tokenizer, bert_model)
    trainer = TEMPTrainer(embedding_provider, 'data/models/TEMP/checkpoints')
    trainer.train(
        model,
        optimizer,
        loss_fn,
        lambda: ds_creator.prepare_ds(train_synsets),
        ds_creator.prepare_ds(test_synsets),
        10,
        'cpu'
    )


run()
