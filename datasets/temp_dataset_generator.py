import random
import time

import torch
from nltk.corpus import WordNetCorpusReader
from nltk.corpus.reader import Synset
from transformers import BertTokenizer, BertModel

import common.performance
from utils.utils import get_synset_simple_name


class TEMPSamplesCreator:
    def __init__(self, wn_reader, tokenizer, bert_model):
        self.__wn_reader = wn_reader
        self.__tokenizer = tokenizer
        self.__bert_model = bert_model

    def __get_negative_sample(self, path, node, all_synsets: [Synset]):
        random_node = random.choice(all_synsets)
        i = 0
        while True:
            random_node_name = get_synset_simple_name(random_node)
            if random_node_name in path:
                random_node = random.choice(all_synsets)
            else:
                break
            i += 1
        print('Chose node in', i, 'iterations')
        return random_node.hypernym_paths()[0].append(node)

    def __get_embeddings_for_samples(self, path):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        path.reverse()
        tokens = [
            path[0].definition(),
            *list(
                map(
                    lambda x: get_synset_simple_name(x),
                    path[1:]
                )
            )
        ]
        encoding = tokenizer.batch_encode_plus(
            tokens,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state
        return word_embeddings

    def __process_node(self, node: Synset, all_synsets: [Synset], negative_samples_size=10):
        path = node.hypernym_paths()[0]
        res = [path]
        for i in range(negative_samples_size):
            res.append(self.__get_negative_sample(path, node, all_synsets))
        return res

    def prepare_ds(self, train_synsets: [Synset]):
        start = time.time()
        all_synsets = self.__wn_reader.all_synsets('n')
        batches = []
        for node in train_synsets:
            samples = self.__process_node(node, all_synsets)
            embeddings = list(map(self.__get_embeddings_for_samples, samples))
            batches.append(embeddings)
        end = time.time()
        print('Finised creating samples in', end - start, 'seconds')
        return batches


wn_reader = WordNetCorpusReader('data/wordnets/WordNet-2.0/dict', None)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
samples_creator = TEMPSamplesCreator(wn_reader, tokenizer, bert_model)
samples_creator.prepare_ds([wn_reader.synset('dog.n.01')])
