import torch
from nltk.corpus import WordNetCorpusReader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider, create_synsets_batch
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.SynsetWrapper import RuSynsetWrapper
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class _TermSynsetAdapter:
    def __init__(self, term):
        self.__term = term

    def name(self):
        return '{}.n.00'.format(self.__term.value)

    def definition(self):
        return self.__term.definition()

    def __str__(self):
        return f'TermSynsetAdapter({self.name()})'


class TEMPTermInferencePerformer:
    def __init__(self, synsets_batch_provider, embedding_provider: TEMPEmbeddingProvider, k=5):
        self.__synsets_batch_provider = synsets_batch_provider
        self.__embedding_provider = embedding_provider
        self.__k = k

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

        for i in range(len(scores_and_paths)):
            scores_and_paths[i] = sorted(scores_and_paths[i], key=lambda x:-x[0])
        return scores_and_paths

    def __update_scores(self, offset, scores, candidate_path, result_buffer):
        for i in range(len(result_buffer)):
            item = result_buffer[i]
            score = scores[offset + i]
            if item is None:
                result_buffer[i] = [(score, candidate_path)]
            elif len(item) < self.__k:
                item.append((score, candidate_path))
            else:
                r = self.__find_min_element_less_then_score(item, score)
                if r > -1:
                    item[r] = (score, candidate_path)

    def __find_min_element_less_then_score(self, scores_and_paths, score) -> int:
        selected_index_score = 1 << 32
        r = -1
        for j in range(len(scores_and_paths)):
            current_score = scores_and_paths[j][0]
            if current_score < score and selected_index_score > current_score:
                r = j
                selected_index_score = current_score
        return r

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
    def create(device, batch_size, wn_reader: WordNetCorpusReader):
        all_synsets = list(wn_reader.all_synsets('n'))
        synsets_batches = create_synsets_batch(all_synsets, batch_size)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        return TEMPTermInferencePerformer(synsets_batches, TEMPEmbeddingProvider(tokenizer, bert_model, device))

    @staticmethod
    def create_ru(device, batch_size, ru_wordnet_session):
        from ruwordnet import ruwordnet
        wn_reader = ruwordnet.RuWordNet(ru_wordnet_session)
        all_synsets = list(map(RuSynsetWrapper, wn_reader.synsets))
        synsets_batches = create_synsets_batch(all_synsets, batch_size)
        tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased').to(device)
        return TEMPTermInferencePerformer(synsets_batches, TEMPEmbeddingProvider(tokenizer, bert_model, device))
