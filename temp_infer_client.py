import argparse

import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import infer_many
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='used device')
args.add_argument('-t', '--terms_path', default=None, type=str, help='terms to infer dataset path')
args.add_argument('-load', '--load-path', default=None, type=str, help='path to saved model')
args.add_argument('-l', '--limit', default=None, type=int, help='terms limit')
args.add_argument('-res', '--result-path', default=None, type=str, help='Result path')
args = args.parse_args()


def read_terms(path, limit):
    with open(path, 'r') as file:
        res = []
        for i in range(limit):
            res.append(file.readline().strip())
        return res


wn_reader = WordNetDao.get_wn_30()
terms = read_terms(args.terms_path, args.limit)

terms = list(map(lambda x: Term(x, wn_reader.synsets(x)[0].definition()), terms))
model = TEMP().to(args.device)
model.load_state_dict(torch.load(args.load_path, map_location=torch.device(args.device)))
delta, results = performance.measure(lambda: infer_many(model, terms, args.device))
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


save(results, args.result_path)
