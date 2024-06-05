import threading
from multiprocessing.pool import ThreadPool

import torch
from transformers import BertConfig

from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_infer import TaxoPromptTermInferencePerformerFactory
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_model import TaxoPrompt
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.SynsetWrapper import RuSynsetWrapper
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import paginate, get_synset_simple_name

device = 'cpu'
terms_path = 'data/datasets/test_temp.tsv'
load_path = 'data/models/TaxoPrompt/taxo_prompt_model_epoch_19'
result_path = 'data/results/TaxoPrompt/predicted_food.tsv'

model = TaxoPrompt(BertConfig()).to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
model.eval()

res_terms = []
wn = WordNetDao.get_wn_30()
test_synsets = set()

with open(terms_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        data = line.strip().split('\t')
        word = data[0]
        test_synsets.add(word)
        synset = wn.synset(word)
        definition = synset.definition()
        res_terms.append(Term(word, definition))
wn_reader = WordNetDao.get_wn_30()

def format_result(scores):
    res = ''
    for c in scores:
        res += f'{c} {scores[c][1]}\n'
    return res


inferer = TaxoPromptTermInferencePerformerFactory.create(device)

def run(terms_batch):
    delta, results = performance.measure(lambda: inferer.infer(model, terms_batch, device))
    print(delta)
    print(results)
    res_str = format_result(results)
    with open(result_path, 'a') as append_file:
        append_file.write(res_str)
    print('Got result for {} terms'.format(len(terms_batch)))

with torch.no_grad():
    batches = paginate(res_terms, 1)
    for batch in batches:
        run(batch)
