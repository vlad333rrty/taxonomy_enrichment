import time

from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.utils.utils import get_synset_simple_name


class TEMPEmbeddingProvider:
    def __init__(self, tokenizer: BertTokenizer, bert_model: BertModel, device):
        self.__tokenizer = tokenizer
        self.__bert_model = bert_model
        self.__device = device

    def get_path_embeddings(self, batch):
        tokens = []
        for path in batch:
            reversed_path = list(reversed(path))
            tokens.append(
                '{} [SEP] {}'.format(reversed_path[0].definition(),
                                     ' '.join(map(lambda x: get_synset_simple_name(x), reversed_path[1:])))
            )

        start = time.time()
        encoding = self.__tokenizer.batch_encode_plus(
            tokens,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        end = time.time()
        print('Encoding finished in', end - start)
        input_ids = encoding['input_ids'].to(self.__device)
        attention_mask = encoding['attention_mask'].to(self.__device)
        outputs = self.__bert_model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        print('Got embedding in', time.time() - end)
        return cls_embedding
