from collections import deque
from typing import Dict, Any

from nltk.corpus.reader import Synset

from src.taxo_expantion_methods.TaxoPrompt.terms_relation import Relation


class ExtendedTaxoGraphNode:
    def __init__(self, synset: Synset):
        self.__synset = synset
        self.__edges = []

    def add_sibling(self, node):
        self.__edges.append((Relation.SIBLING_OF, node))

    def add_parent(self, node):
        self.__edges.append((Relation.CHILD_OF, node))

    def add_child(self, node):
        self.__edges.append((Relation.PARENT_OF, node))

    def get_synset(self):
        return self.__synset

    def get_edges(self):
        return self.__edges

    def get_parent_nodes(self):
        return list(
            map(
                lambda x: x[1],
                filter(
                    lambda x: x[0] == Relation.CHILD_OF,
                    self.__edges
                )
            )

        )

    def __str__(self):
        return f'ExtendedTaxoGraphNode({self.__synset.name()})'

    def __hash__(self):
        return self.__synset.__hash__()

    def __eq__(self, other):
        return self.__synset == other.__synset and self.__edges == other.__edges


def create_extended_taxo_graph(taxo_root: Synset):
    queue = deque()
    root_node = ExtendedTaxoGraphNode(taxo_root)
    queue.append(root_node)
    synset_to_node = {taxo_root: root_node}
    while len(queue) > 0:
        all_siblings = []
        while len(queue) > 0:
            all_siblings.append(queue.popleft())
        for i in range(len(all_siblings) - 1):
            u = all_siblings[i]
            v = all_siblings[i + 1]
            u.add_sibling(v)
            v.add_sibling(u)

        for elem in all_siblings:
            children = elem.get_synset().hyponyms()
            for child in children:
                if child in synset_to_node:
                    node = synset_to_node[child]
                else:
                    node = ExtendedTaxoGraphNode(child)
                    queue.append(node)
                    synset_to_node[child] = node
                elem.add_child(node)
                node.add_parent(elem)

    return synset_to_node
