class PrecalcEmbeddingsProvider:
    def __init__(self, embeddings_graph):
        self.__embeddings_graph = embeddings_graph


    def get_embeddings(self, synset):
        return self.__embeddings_graph[synset.name()]

