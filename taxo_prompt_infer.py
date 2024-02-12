import threading
from multiprocessing.pool import ThreadPool

import torch
from transformers import BertConfig

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_infer import TaxoPromptTermInferencePerformerFactory
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_model import TaxoPrompt
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import paginate

device = 'cpu'
terms_path = 'data/datasets/diachronic-wordnets/en/no_labels_nouns_en.2.0-3.0.tsv'
load_path = 'data/models/TaxoPrompt/pre-trained/taxo_prompt_model_epoch_10'
result_path = 'data/results/TaxoPrompt/predicted.tsv'
limit = 10

model = TaxoPrompt(BertConfig()).to(device)
model.load_state_dict(
    torch.load('data/models/TaxoPrompt/pre-trained/taxo_prompt_model_epoch_15', map_location=torch.device(device)))


def read_terms(path, _limit):
    with open(path, 'r') as _file:
        res = []
        for i in range(_limit):
            res.append(_file.readline().strip())
        return res


wn_reader = WordNetDao.get_wn_30()

terms = read_terms(terms_path, limit)
res_terms = []
for term in terms:
    synsets = wn_reader.synsets(term)
    res_terms += list(map(lambda x: Term(term, x.definition()), synsets))


def format_result(scores):
    res = ''
    for c in scores:
        res += f'{c} {scores[c][1]}\n'
    return res


inferer = TaxoPromptTermInferencePerformerFactory.create(device, 16)

file_write_lock = threading.Lock()


def run(terms_batch):
    concepts = list(map(lambda x: x.value, terms_batch))
    defs = list(map(lambda x: x.definition, terms_batch))
    delta, results = performance.measure(lambda: inferer.infer(model, concepts, defs, device))
    print(delta)
    print(results)
    res_str = format_result(results)
    file_write_lock.acquire()
    with open(result_path, 'a') as append_file:
        append_file.write(res_str)
    file_write_lock.release()
    print('Got result for {} terms'.format(len(terms_batch)))


pool = ThreadPool(1)
batches = paginate(res_terms, 2)
pool.map(run, batches)
