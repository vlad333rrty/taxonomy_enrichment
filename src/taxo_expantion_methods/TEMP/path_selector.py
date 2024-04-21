import random


class WnPathSelector:
    def select_path(self, node):
        paths = node.hypernym_paths()
        return random.choice(list(
            filter(
                lambda x: x[0].name() == 'entity.n.01',
                paths
            ))
        )


class RuWnPathSelector:
    def select_path(self, node):
        paths = node.hypernym_paths()
        return random.choice(paths)