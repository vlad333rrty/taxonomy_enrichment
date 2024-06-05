from sklearn.model_selection import train_test_split

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


def gen(root_synset, train_path, test_path):
    leaf_synsets = SynsetsProvider.get_all_leaf_synsets_with_common_root(root_synset)
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(root_synset)
    print('Got {} synsets and {} leaf synsets'.format(len(all_synsets), len(leaf_synsets)))

    train_synsets, test_synsets = train_test_split(leaf_synsets, train_size=0.8, test_size=0.2)
    with open(test_path, 'w') as file:
        res_str = ''
        for s in test_synsets:
            res_str += '{}\t{}\n'.format(s.name(), ','.join(list(map(lambda x: x.name(), s.hypernyms()))))
        file.write(res_str)

    with open(train_path, 'w') as file:
        res_str = ''
        for s in train_synsets:
            res_str += '{}\n'.format(s.name())
        file.write(res_str)


wn = WordNetDao.get_wn_30()
root = wn.synset('food.n.01')
gen(root, 'data/datasets/temp_train.tsv', 'data/datasets/temp_test.tsv')