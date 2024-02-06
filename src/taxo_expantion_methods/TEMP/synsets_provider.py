from nltk.corpus.reader import Synset


class SynsetsProvider:
    @staticmethod
    def __dfs(root: Synset):
        stack = [root]
        used = set()
        leafs = []
        while len(stack) > 0:
            u = stack.pop()
            if len(u.hyponyms()) == 0:
                leafs.append(u)
            for child in u.hyponyms():
                if child not in used:
                    stack.append(child)
            used.add(u)
        return leafs

    @staticmethod
    def get_all_synsets_with_common_root(root: Synset):
        return SynsetsProvider.__dfs(root)