import torch

from src.taxo_expantion_methods.TEMP.client.temp_infer import TEMPTermInferencePerformerFactory
from src.taxo_expantion_methods.TEMP.path_selector import SubgraphPathSelector
from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddDataParser
from src.taxo_expantion_methods.utils.utils import paginate

device = 'cpu'
terms_path = 'data/datasets/test_temp.tsv'
load_path = 'data/models/TEMP/pre-trained/temp_model_epoch_49'
result_path = 'data/results/TEMP/predicted_food.tsv'

res_terms = []
wn = WordNetDao.get_wn_30()
test_synsets = set()

with open(terms_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        data = line.strip().split('\t')
        word = data[0]
        test_synsets.add(word)
        definition = wn.synset(word).definition()
        res_terms.append(Term(word, definition))

model = TEMP().to(device)
model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))




root = wn.synset('food.n.01')
all_synsets = SynsetsProvider.get_all_synsets_with_common_root(root)
all_synsets = list(
    filter(
        lambda s: s.name() not in test_synsets,
        all_synsets
    )
)

class PathFilter:
    def select_path(self, node):
        paths = node.hypernym_paths()
        selected_paths = list(filter(lambda p: root in p, paths))
        result_paths = []
        for path in selected_paths:
            i = 0
            while path[i] != root:
                i += 1
            result_paths.append(path[i:])
        return result_paths


inference_performer = TEMPTermInferencePerformerFactory.create(device, 16, all_synsets, PathFilter())


def format_result(_terms, results):
    res_str = ''
    for i in range(len(_terms)):
        result = results[i]
        term = _terms[i]
        anchors = []
        for r in result:
            path = r[1]
            anchors.append(path[-1].name())

        res_str += '{}\t{}\n'.format(term.value(), ','.join(anchors))
    return res_str


def run(terms_batch):
    delta, results = performance.measure(lambda: inference_performer.infer(model, terms_batch))
    print(delta)
    print(results)
    res_str = format_result(terms_batch, results)
    with open(result_path, 'a') as append_file:
        append_file.write(res_str)
    print('Got result for {} terms'.format(len(terms_batch)))


with torch.no_grad():
    batches = paginate(res_terms, 1)
    for batch in batches:
        run(batch)
