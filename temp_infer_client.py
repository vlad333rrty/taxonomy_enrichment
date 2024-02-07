import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformerFactory
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao

device = 'cpu'
terms_path = 'data/datasets/diachronic-wordnets/en/no_labels_nouns_en.2.0-3.0.tsv'
load_path = 'data/models/TEMP/pre-trained/temp_model_epoch_5'
result_path = 'data/results/TEMP/predicted.tsv'
limit = 1


def read_terms(path, limit):
    with open(path, 'r') as file:
        res = []
        for i in range(limit):
            res.append(file.readline().strip())
        return res


wn_reader = WordNetDao.get_wn_30()
terms = read_terms(terms_path, limit)

terms = list(map(lambda x: Term(x, wn_reader.synsets(x)[0].definition()), terms))
model = TEMP().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))

inference_performer = TEMPTermInferencePerformerFactory.create(device, 16)

delta, results = performance.measure(lambda: inference_performer.infer(model, terms))
print(delta)
print(results)


def save(results, path):
    res_str = ''
    for r in results:
        if len(r) > 2:
            res_str += '{} {},{}\n'.format(r[-1], r[-2], r[-3])
        else:
            res_str += '{} {}\n'.format(r[-1], r[-2])
    with open(path, 'w') as file:
        file.write(res_str)


save(results, result_path)
