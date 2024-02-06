from multiprocessing.pool import ThreadPool

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

    def infer(self, model: TEMP, terms_batch: [Term]):
        term_sysnet_adapters = list(map(lambda term:_TermSynsetAdapter(term), terms_batch))
        scores_and_paths = [None for _ in terms_batch]
        with torch.no_grad():
            for synset in tqdm(self.__all_synsets):
                paths = synset.hypernym_paths()
                for path in paths:
                    candidate_paths = list(map(lambda t: path + [t], term_sysnet_adapters))
                    current_scores = self.__get_scores_for_path(model, candidate_paths)
                    for i in range(len(scores_and_paths)):
                        item = scores_and_paths[i]
                        score = current_scores[i]
                        if item is None:
                            scores_and_paths[i] = (score, candidate_paths[i])
                        elif item[0] < score:
                            scores_and_paths[i] = (score, candidate_paths[i])
        return scores_and_paths

    def __get_scores_for_path(self, model, paths):
        embeddings = self.__embedding_provider.get_path_embeddings(paths)
        return model(embeddings)


def infer_many(model, terms, device):
    wn_reader = WordNetDao.get_wn_20()
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    inference_performer = TEMPTermInferencePerformer(all_synsets, TEMPEmbeddingProvider(tokenizer, bert_model, device))
    result = inference_performer.infer(model, terms)
    return result

def infer_many_async(model, terms, device, workers):
    wn_reader = WordNetDao.get_wn_20()
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    inference_performer = TEMPTermInferencePerformer(all_synsets, TEMPEmbeddingProvider(tokenizer, bert_model, device))
    pool = ThreadPool(workers)
    result = pool.map(lambda term: inference_performer.infer(model, term), terms)
    return result