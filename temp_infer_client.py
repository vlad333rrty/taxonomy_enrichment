from multiprocessing.pool import ThreadPool

import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformerFactory
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import paginate

device = 'cpu'
terms_path = 'data/datasets/diachronic-wordnets/en/no_labels_nouns_en.2.0-3.0.tsv'
load_path = 'data/models/TEMP/pre-trained/temp_model_epoch_8'
result_path = 'data/results/TEMP/predicted.tsv'
limit = 8


def read_terms(path, _limit):
    with open(path, 'r') as _file:
        res = []
        for i in range(_limit):
            res.append(_file.readline().strip())
        return res


wn_reader = WordNetDao.get_wn_30()
terms = read_terms(terms_path, limit)

terms = list(map(lambda x: Term(x, wn_reader.synsets(x)[0].definition()), terms))
model = TEMP().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))

inference_performer = TEMPTermInferencePerformerFactory.create(device, 16)


def run(terms_batch):
    delta, results = performance.measure(lambda: inference_performer.infer(model, terms))
    print(delta)
    print(results)
    result = list(map(lambda x: x[1], results))
    return terms_batch, result


pool = ThreadPool(4)
batches = paginate(terms, 2)
all_results = pool.map(run, batches)


def format_result(_terms, results):
    res_str = ''
    for i in range(len(_terms)):
        result = results[i]
        term = _terms[i]
        if len(result) > 2:
            res_str += '{} {},{}\n'.format(term.value, result[-1].name(), result[-2].name())
        else:
            res_str += '{} {}\n'.format(term.value, result[-1].name())
    return res_str


formatted = ''.join(list(
    map(
        lambda x: format_result(x[0], x[1]),
        all_results
    )
))

with open(result_path, 'w') as file:
    file.write(formatted)
