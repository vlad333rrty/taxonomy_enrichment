import threading
from multiprocessing.pool import ThreadPool

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_infer import infer
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import paginate

device = 'cpu'
terms_path = 'data/datasets/diachronic-wordnets/en/no_labels_nouns_en.2.0-3.0.tsv'
load_path = 'data/models/TaxoPrompt/pre-trained/taxo_prompt_model_epoch_15'
result_path = 'data/results/TaxoPrompt/predicted.tsv'
limit = 10


def read_terms(path, _limit):
    with open(path, 'r') as _file:
        res = []
        for i in range(_limit):
            res.append(_file.readline().strip())
        return res
wn_reader = WordNetDao.get_wn_30()
all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))

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

file_write_lock = threading.Lock()


def run(terms_batch):
    concepts = list(map(lambda x:x.value, terms_batch))
    defs = list(map(lambda x:x.definition, terms_batch))
    delta, results = performance.measure(lambda: infer(device, concepts, defs, all_synsets))
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
