from ruwordnet.models import Synset


class RuSynsetWrapper: # adapter?
    def __init__(self, synset: Synset):
        self.__synset = synset

    def hypernym_paths(self):
        candidates = []
        stack = []
        stack.append((self.__synset, 0, [RuSynsetWrapper(self.__synset)]))
        used = set()
        while len(stack) > 0:
            u_h = stack.pop()
            u, h = u_h[0], u_h[1]
            parents = u.hypernyms
            if len(parents) == 0:
                candidates.append((u_h[2], h))
            for parent in parents:
                if parent not in used:
                    stack.append((parent, h + 1, u_h[2] + [RuSynsetWrapper(parent)]))
            used.add(u)

        return list(
            map(
                lambda x: list(reversed(x[0])),
                sorted(candidates, key=lambda x: -x[1])
            )
        )

    def name(self):
        return self.__synset.title

    def hypernyms(self):
        return list(map(RuSynsetWrapper, self.__synset.hypernyms))

    def definition(self):
        definition = self.__synset.definition
        return definition

    def __repr__(self):
        return self.__synset.__repr__()

    def id(self):
        return self.__synset.id
