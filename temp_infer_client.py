import threading

import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformerFactory
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddDataParser
from src.taxo_expantion_methods.utils.utils import paginate

device = 'cpu'
terms_path = 'data/datasets/semeval/training_data.csv'
load_path = 'data/models/TEMP/pre-trained/temp_model_epoch_7'
result_path = 'data/results/TEMP/predicted2.tsv'
limit = 1

terms = WordToAddDataParser.from_pandas(terms_path, '$')

res_terms = []
for term in terms:
    res_terms.append(Term(term.value, term.definition))

model = TEMP().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))

inference_performer = TEMPTermInferencePerformerFactory.create(device, 16, WordNetDao.get_wn_30())


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
