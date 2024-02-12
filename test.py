from src.taxo_expantion_methods.common.wn_dao import WordNetDao

wn = WordNetDao.get_wn_30()
s = wn.synset('benzodiazepine.n.01')
print(s.hypernym_paths())
print(s.definition())
t = wn.synset('virility_drug.n.01')
a = s.lowest_common_hypernyms(t)

print(t.hypernym_paths())