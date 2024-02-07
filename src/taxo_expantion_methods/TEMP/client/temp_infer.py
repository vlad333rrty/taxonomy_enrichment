import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider, create_synsets_batch
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
    def __init__(self, synsets_batch_provider, embedding_provider: TEMPEmbeddingProvider):
        self.__synsets_batch_provider = synsets_batch_provider
        self.__embedding_provider = embedding_provider

    def infer(self, model: TEMP, terms_batch: [Term]):
        term_sysnet_adapters = list(map(lambda term: _TermSynsetAdapter(term), terms_batch))
        scores_and_paths = [None for _ in terms_batch]
        with torch.no_grad():
            for synset in tqdm(self.__synsets_batch_provider):
                paths = synset.hypernym_paths()
                candidate_paths = self.__get_candidates_paths(paths, term_sysnet_adapters)
                current_scores = self.__get_scores_for_path(model, candidate_paths)
                for i in range(len(paths)):
                    self.__update_scores(i * len(term_sysnet_adapters), current_scores, paths[i],
                                         scores_and_paths)
        return scores_and_paths

    def __update_scores(self, offset, scores, candidate_path, result_buffer):
        for i in range(len(result_buffer)):
            item = result_buffer[i]
            score = scores[offset + i]
            if item is None:
                result_buffer[i] = (score, candidate_path)
            elif item[0] < score:
                result_buffer[i] = (score, candidate_path)

    def __get_candidates_paths(self, taxonomy_paths, terms):
        candidate_paths = []
        for path in taxonomy_paths:
            candidate_paths += list(map(lambda t: path + [t], terms))
        return candidate_paths

    def __get_scores_for_path(self, model, paths):
        embeddings = self.__embedding_provider.get_path_embeddings(paths)
        return model(embeddings)


class TEMPTermInferencePerformerFactory:
    @staticmethod
    def create(device, batch_size):
        wn_reader = WordNetDao.get_wn_20()
        all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))[:100]
        synsets_batches = create_synsets_batch(all_synsets, batch_size)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        return TEMPTermInferencePerformer(synsets_batches, TEMPEmbeddingProvider(tokenizer, bert_model, device))
