import torch
from nltk.corpus.reader import Synset
from transformers import BertModel, BertTokenizer

from src.taxo_expantion_methods.parent_sum.embeddings_graph import EmbeddingsGraphNode, NodeEmbeddings


class IsAEmbeddingsProvider:
    def __init__(self, embeddings_graph, bert_model: BertModel, tokenizer: BertTokenizer, device):
        self.__embeddings_graph = embeddings_graph
        self.__bert_model = bert_model
        self.__tokenizer = tokenizer
        self.__device = device


    def fallback(self, synset: Synset):
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
        return cls_embedding

    def __get_inner(self, synset):
        key = synset.name()
        embed = self.__embeddings_graph.get(key)
        if embed is None:
            embed = self.fallback(synset)
            self.__embeddings_graph[key] = EmbeddingsGraphNode(key, [NodeEmbeddings(embed, None, 0)])
        else:
            embed = embed.get_embeddings()[0].embedding
        return embed

    def get_embeddings(self, synset_pairs):
        res = []
        with torch.no_grad():
            for synset_pair in synset_pairs:
                embed1 = self.__get_inner(synset_pair[0])
                embed2 = self.__get_inner(synset_pair[1])
                res.append(torch.cat([embed1, embed2], dim=1))
        return torch.cat(res)
