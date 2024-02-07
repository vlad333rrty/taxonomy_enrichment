import unittest

from src.taxo_expantion_methods.utils.utils import paginate


class PaginationTest(unittest.TestCase):
    def test_page_size_divides_array_len(self):
        array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        pages = paginate(array, 3)
        self.assertTrue(len(pages) == 3)
        self.assertEquals([1, 2, 3], pages[0])
        self.assertEquals([4, 5, 6], pages[1])
        self.assertEquals([7, 8, 9], pages[2])

    def test_page_size_doesnt_divide_array_len(self):
        array = [1, 2, 3, 4, 5]
        pages = paginate(array, 2)
        self.assertTrue(len(pages) == 3)
        self.assertEquals([1, 2], pages[0])
        self.assertEquals([3, 4], pages[1])
        self.assertEquals([5], pages[2])