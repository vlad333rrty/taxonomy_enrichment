from abc import ABC, abstractmethod

from nltk.corpus.reader import Synset
from transformers import BertTokenizer, BertModel


class EmbeddingsGraphNodeEmbeddingsProvider(ABC):
    @abstractmethod
    def get_embedding(self, synset: Synset):
        raise NotImplementedError()


class EmbeddingsGraphBERTNodeEmbeddingsProvider(EmbeddingsGraphNodeEmbeddingsProvider):
    def __init__(self, bert_model: BertModel, tokenizer: BertTokenizer, device):
        self.__bert_model = bert_model
        self.__tokenizer = tokenizer
        self.__device = device

    def get_embedding(self, synset: Synset):
        encoding = self.__tokenizer.encode_plus(
            synset.definition(),
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = encoding['input_ids'].to(self.__device)
        attention_mask = encoding['attention_mask'].to(self.__device)
        outputs = self.__bert_model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.numpy()
