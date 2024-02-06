import argparse

import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import infer
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='used device')
args.add_argument('-t', '--term', default=None, type=int, help='term to infer')
args.add_argument('-l', '--load-path', default=None, type=str, help='path to saved model')
args = args.parse_args()

wn_reader = WordNetDao.get_wn_30()
term = args.term

model = TEMP(768 * 2, 256)
model.load_state_dict(torch.load(args.load_path, map_location=torch.device(args.device)))

delta, result = performance.measure(
    lambda: infer(model, Term(term, wn_reader.synsets(term)[0].definition()), args.device))

print(delta)
print(result)
