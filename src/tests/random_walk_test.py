import unittest

from src.taxo_expantion_methods.TaxoPrompt.random_walk import create_extended_taxo_graph
from src.taxo_expantion_methods.TaxoPrompt.terms_relation import Relation


class RandomWalkTest(unittest.TestCase):
    class _SynsetMock:
        def __init__(self, name, hyponyms):
            self.__name = name
            self.__hyponyms = hyponyms

        def name(self):
            return self.__name

        def hyponyms(self):
            return self.__hyponyms

        def __hash__(self):
            return self.__name.__hash__()

        def __eq__(self, other):
            return self.__name == other.__name

    def test_root_has_2_leaf_children(self):
        child2 = self._SynsetMock('child2', [])
        child1 = self._SynsetMock('child1', [])
        root = self._SynsetMock('root', [child1, child2])
        synset2node = create_extended_taxo_graph(root)
        root_node = synset2node[root]

        self.assertEquals(root_node.get_synset(), root)

        edge_to_child1 = root_node.get_edges()[0]
        self.assertEquals(edge_to_child1[0], Relation.PARENT_OF)
        self.assertEquals(edge_to_child1[1].get_synset(), child1)

        edge_to_child2 = root_node.get_edges()[0]
        self.assertEquals(edge_to_child2[0], Relation.PARENT_OF)
        self.assertEquals(edge_to_child2[1].get_synset(), child1)

        # check child 1
        self.assertEquals(2, len(edge_to_child1[1].get_edges()))

        child1_edges = edge_to_child1[1].get_edges()
        back_edges = list(filter(lambda x: x[0] == Relation.CHILD_OF, child1_edges))
        sibling_edges = list(filter(lambda x: x[0] == Relation.SIBLING_OF, child1_edges))

        back_edge1 = back_edges[0]
        self.assertEquals(Relation.CHILD_OF, back_edge1[0])
        self.assertEquals(root_node, back_edge1[1])

        sibling_edge1 = sibling_edges[0]
        self.assertEquals(Relation.SIBLING_OF, sibling_edge1[0])
        self.assertEquals(child2, sibling_edge1[1].get_synset())

        # check child 2
        self.assertEquals(2, len(edge_to_child2[1].get_edges()))

        child2_edges = edge_to_child2[1].get_edges()
        back_edges = list(filter(lambda x: x[0] == Relation.CHILD_OF, child2_edges))
        sibling_edges = list(filter(lambda x: x[0] == Relation.SIBLING_OF, child2_edges))

        back_edge2 = back_edges[0]
        self.assertEquals(Relation.CHILD_OF, back_edge2[0])
        self.assertEquals(root_node, back_edge2[1])

        sibling_edge2 = sibling_edges[0]
        self.assertEquals(Relation.SIBLING_OF, sibling_edge2[0])
        self.assertEquals(child2, sibling_edge2[1].get_synset())

    def test_root_has_2_children_with_common_child(self):
        grand_child1 = self._SynsetMock('grand_child1', [])
        child2 = self._SynsetMock('child2', [grand_child1])
        child1 = self._SynsetMock('child1', [grand_child1])
        root = self._SynsetMock('root', [child1, child2])
        synset2node = create_extended_taxo_graph(root)
        root_node = synset2node[root]

        child1_edges = root_node.get_edges()[0][1].get_edges()
        child2_edges = root_node.get_edges()[1][1].get_edges()

        child1_parent_of = list(filter(lambda x: x[0] == Relation.PARENT_OF, child1_edges))
        child2_parent_of = list(filter(lambda x: x[0] == Relation.PARENT_OF, child2_edges))
        self.assertEquals(child1_parent_of, child2_parent_of)

        self.assertEquals(2, len(child1_parent_of[0][1].get_edges()))