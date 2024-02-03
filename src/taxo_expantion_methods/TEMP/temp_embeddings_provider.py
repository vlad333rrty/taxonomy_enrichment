import time

from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.utils.utils import get_synset_simple_name


class TEMPEmbeddingProvider:
    def __init__(self, tokenizer: BertTokenizer, bert_model: BertModel):
        self.__tokenizer = tokenizer
        self.__bert_model = bert_model

    def get_path_embedding(self, path):
        reversed_path = list(reversed(path))
        tokens = [
            reversed_path[0].definition(),
            ' '.join(
                map(
                    lambda x: get_synset_simple_name(x),
                    reversed_path[1:]
                )
            )
        ]

        encoding = self.__tokenizer.batch_encode_plus(
            tokens,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.__bert_model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding[0]
