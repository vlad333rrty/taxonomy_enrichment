import torch
from nltk.corpus import WordNetCorpusReader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.is_a.IsAClassifier import IsALoss
from src.taxo_expantion_methods.is_a.graph_embedding_provider import IsAEmbeddingsProvider
from src.taxo_expantion_methods.is_a.is_a_dataset_generator import IsADatasetGenerator
from src.taxo_expantion_methods.is_a.train import IsATrainer


def run_isa_model_training(embeddings_graph, device, epochs, res_path, model, wn_reader: WordNetCorpusReader, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    loss_fn = IsALoss(batch_size, device)
    all_synsets = list(wn_reader.all_synsets('food.n.01'))
    print('Got {} synsets overall'.format(len(all_synsets)))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    train_synsets = []
    with open('data/datasets/temp_train.tsv', 'r') as file:
        lines = file.readlines()
    for line in lines:
        train_synsets.append(wn_reader.synset(line.strip()))

    print('Got {} train synsets'.format(len(train_synsets)))

    ds_creator = IsADatasetGenerator(all_synsets)
    embedding_provider = IsAEmbeddingsProvider(embeddings_graph, bert_model, tokenizer, device)
    trainer = IsATrainer(embedding_provider, res_path)
    trainer.train(
        model,
        optimizer,
        loss_fn,
        lambda: ds_creator.generate(train_synsets, batch_size),
        epochs
    )

