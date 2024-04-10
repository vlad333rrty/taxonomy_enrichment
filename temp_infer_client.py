import threading
from multiprocessing.pool import ThreadPool

import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformerFactory
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance, RuWordNetDao
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import paginate

device = 'cpu'
terms_path = 'data/datasets/diachronic-wordnets/ru/nouns_public_no_labels.tsv'
load_path = 'data/models/TEMP/pre-trained/temp_model_ru'
result_path = 'data/results/TEMP/predicted_ru.tsv'
limit = 100


def read_terms(path, _limit):
    with open(path, 'r') as _file:
        res = []
        for i in range(_limit):
            res.append(_file.readline().strip())
        return res


wn_reader = RuWordNetDao.get_ru_wn_21()
terms = read_terms(terms_path, limit)

res_terms = []
for term in terms:
    synsets = wn_reader.get_synsets(term)
    res_terms += list(map(lambda x: Term(term, x.definition), synsets))

model = TEMP().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))

inference_performer = TEMPTermInferencePerformerFactory.create(device, 16)


def format_result(_terms, results):
    res_str = ''
    for i in range(len(_terms)):
        result = results[i]
        term = _terms[i]
        anchors = []
        for r in result:
            path = r[1]
            anchors.append(path[-1].name())
            if len(path) > 2:
                anchors.append(path[-2].name())

        res_str += '{}\t{}\n'.format(term.value, ','.join(anchors))
    return res_str


file_write_lock = threading.Lock()


def run(terms_batch):
    delta, results = performance.measure(lambda: inference_performer.infer(model, terms_batch))
    print(delta)
    print(results)
    res_str = format_result(terms_batch, results)
    file_write_lock.acquire()
    with open(result_path, 'a') as append_file:
        append_file.write(res_str)
    file_write_lock.release()
    print('Got result for {} terms'.format(len(terms_batch)))



with torch.no_grad():
  batches = paginate(res_terms, 1)
  for batch in batches:
    run(batch)
