import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class _TermSynsetAdapter:
    def __init__(self, term):
        self.__term = term

    def name(self):
        return '{}.n.00'.format(self.__term.value)

    def definition(self):
        return self.__term.definition

    def __str__(self):
        return f'TermSynsetAdapter({self.name()})'


class TEMPTermInferencePerformer:
    def __init__(self, all_synsets, embedding_provider: TEMPEmbeddingProvider):
        self.__all_synsets = all_synsets
        self.__embedding_provider = embedding_provider

    def infer(self, model: TEMP, term: Term):
        max_score = -1
        best_item = None
        term_sysnet_adapter = _TermSynsetAdapter(term)
        with torch.no_grad():
            for synset in tqdm(self.__all_synsets):
                paths = synset.hypernym_paths()
                for path in paths:
                    candidate_path = path + [term_sysnet_adapter]
                    current_score = self.__get_score_for_path(model, candidate_path)
                    if current_score > max_score:
                        max_score = current_score
                        best_item = candidate_path
        return best_item

    def __get_score_for_path(self, model, path):
        embedding = self.__embedding_provider.get_path_embedding(path)
        return model(embedding)


def infer(model, term, device):
    wn_reader = WordNetDao.get_wn_20()
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    inference_performer = TEMPTermInferencePerformer(all_synsets, TEMPEmbeddingProvider(tokenizer, bert_model, device))
    result = inference_performer.infer(model, term)
    return result
