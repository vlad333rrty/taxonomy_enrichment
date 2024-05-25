from collections import deque

from nltk.corpus.reader import Synset
from torch import Tensor

from src.taxo_expantion_methods.parent_sum.node_embeddings_provider import EmbeddingsGraphNodeEmbeddingsProvider


class NodeEmbeddings:
    def __init__(self, embedding: Tensor, parent, index: int):
        """
        embedding - embedding tensor
        parent - pointer to parent node
        index = parent embedding index in embeddings list
        """
        self.embedding = embedding
        self.parent = parent
        self.index = index

class EmbeddingsGraphNode:
    def __init__(self, synset, embeddings):
        self.__synset = synset
        self.__embeddings = embeddings

    def get_synset(self):
        return self.__synset

    def get_embeddings(self) -> [NodeEmbeddings]:
        return self.__embeddings


class EmbeddingsGraphProvider:
    def __init__(self, embeddings_provider: EmbeddingsGraphNodeEmbeddingsProvider):
        self.__embeddings_provider = embeddings_provider

    def __create_embeddings_node(self, synset2node, node: Synset):
        parent_nodes: [EmbeddingsGraphNode] = list(
            map(
                synset2node.get,
                node.hypernyms()
            )
        )
        embeddings_accum = []
        for parent in parent_nodes:
            embeddings: [NodeEmbeddings] = parent.get_embeddings()
            for i in range(len(embeddings)):
                embeddings_accum.append(
                    NodeEmbeddings(
                        embeddings[i].embedding + self.__embeddings_provider.get_embedding(node),
                        parent,
                        i
                    )
                )
        return EmbeddingsGraphNode(node, embeddings_accum)

    def from_wordnet(self, root: Synset):
        synset2node = {}
        queue = deque()
        queue.append(root)
        while len(queue) > 0:
            u: Synset = queue.popleft()
            if len(u.hypernyms()) == 0:
                embeddings_node = EmbeddingsGraphNode(u, [NodeEmbeddings(self.__embeddings_provider.get_embedding(u), None, -1)])
            else:
                if len(list(filter(lambda x: x not in synset2node, u.hypernyms()))) > 0:
                    queue.append(u)
                    continue
                embeddings_node = self.__create_embeddings_node(synset2node, u)
            synset2node[u] = embeddings_node
            for child in u.hyponyms():
                if child not in synset2node:
                    queue.append(child)
            print(len(synset2node), end='\r')
        return synset2node

