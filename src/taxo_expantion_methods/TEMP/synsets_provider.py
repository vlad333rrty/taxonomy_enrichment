from nltk.corpus.reader import Synset

from src.taxo_expantion_methods.utils.utils import paginate


class SynsetsProvider:
    @staticmethod
    def __dfs(root: Synset):
        stack = [root]
        used = set()
        result = []
        while len(stack) > 0:
            u = stack.pop()
            result.append(u)
            for child in u.hyponyms():
                if child not in used:
                    stack.append(child)
            used.add(u)
        return result

    @staticmethod
    def get_all_leaf_synsets_with_common_root(root: Synset):
        synsets = SynsetsProvider.get_all_synsets_with_common_root(root)
        return list(filter(lambda s: len(s.hyponyms()) == 0, synsets))

    @staticmethod
    def get_all_synsets_with_common_root(root: Synset):
        return SynsetsProvider.__dfs(root)


class SynsetBatch:
    def __init__(self, synsets):
        self.__synsets = synsets

    def hypernym_paths(self):
        result = []
        for synset in self.__synsets:
            result += synset.hypernym_paths()
        return result


def create_synsets_batch(synsets, batch_size) -> [SynsetBatch]:
    batches = paginate(synsets, batch_size)
    return list(map(SynsetBatch, batches))
