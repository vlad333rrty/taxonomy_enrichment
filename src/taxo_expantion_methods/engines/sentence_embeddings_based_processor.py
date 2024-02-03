import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from nltk.corpus.reader import Synset
from nltk.corpus import wordnet31 as wordnet

from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.set_m import SetM
from src.taxo_expantion_methods.engines.essential_words_gathering_utils import gather_essential_words
from src.taxo_expantion_methods.engines.processor_base import ProcessingResult
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddData
from src.taxo_expantion_methods.utils.similarity import cos_sim
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingsBasedProcessor:
    def __init__(self, cache_name='synset_name_to_embedding', use_cache=True):
        self._cache_name = cache_name
        self.use_cache = use_cache
        self.__closest_sysnets_embeddings_aggregator = None

    def set_closest_sysnets_embeddings_aggregator(self, closest_sysnets_embeddings_aggregator):
        self.__closest_sysnets_embeddings_aggregator = closest_sysnets_embeddings_aggregator

    def process(self, words_to_add_data: [WordToAddData], model_name: str):
        sentence_transformer = SentenceTransformer(model_name)

        sentences = list(
            map(lambda word2add_data: word2add_data.definition, words_to_add_data)
        )

        embeddings = sentence_transformer.encode(sentences)

        words = list(
            map(lambda word2add_data: word2add_data.num, words_to_add_data)
        )

        word2embedding = dict(zip(words, embeddings))

        essential_words_per_word, num2word = gather_essential_words(words_to_add_data, 10)

        word2synsets = self.__get_word2synsets(essential_words_per_word)
        delta, synset_name2embeddings = performance.measure(
            lambda: self.__get_synset_name_to_embedding(word2synsets, sentence_transformer))

        print('Got sentences embeddings in {}s'.format(delta))

        result = []
        counter = 0
        for word_id in word2synsets:
            synsets = word2synsets[word_id]
            closest_synsets = self.__get_k_nearest_synsets(word2embedding[word_id], synsets, synset_name2embeddings, 5)

            if self.__closest_sysnets_embeddings_aggregator is not None:
                self.__closest_sysnets_embeddings_aggregator.aggregate_and_insert_into_table(num2word[word_id], closest_synsets)

            result.append(ProcessingResult(num2word[word_id], word_id, closest_synsets,  'attach'))

            counter += 1
            print('{}% completed'.format(counter / len(essential_words_per_word)))

        return result

    def _construct_candidates_definition_sentences(self, synsets: [Synset]):
        return list(map(Synset.definition, synsets))

    def __get_word2synsets(self,essential_words_per_word, ):
        word2synsets = {}
        for word_id in essential_words_per_word:
            essential_words = essential_words_per_word[word_id][0]
            pos = SentenceEmbeddingsBasedProcessor.__get_appropriate_pos(essential_words_per_word[word_id][1])
            if len(essential_words) > 0:
                synsets = SentenceEmbeddingsBasedProcessor.__get_synsets(essential_words, pos)
                if len(synsets) == 0:
                    print('No synsets found for {} with POS {}'.format(word_id, pos))
                    continue
                word2synsets[word_id] = synsets
            else:
                print('No essential words find for {}'.format(word_id))

        return word2synsets

    def __get_synset_name_to_embedding(self, word2synsets, sentence_transformer):
        is_serialized = os.path.isfile('serialized/' + self._cache_name)
        if is_serialized and self.use_cache:
            print('Taking embeddings from cache...')
            return self.__deserialize_synset_name_to_embedding()

        executor = ThreadPoolExecutor(max_workers=10)
        word2future_and_synsets = {}
        for word in word2synsets:
            synsets = word2synsets[word]
            definitions = self._construct_candidates_definition_sentences(synsets)
            word2future_and_synsets[word] = \
                (executor.submit(
                    (lambda definitions: lambda: sentence_transformer.encode(definitions))(definitions)), synsets,
                 (lambda definitions: lambda: definitions)(definitions))

        synset_name2embeddings = {}
        for word in word2future_and_synsets:
            embeddings = word2future_and_synsets[word][0].result()
            synsets = word2future_and_synsets[word][1]
            if len(synsets) != len(embeddings):
                print('Word: {}, definitions: {}'.format(word, word2future_and_synsets[word][2]()))
                exit(0)
            for i in range(len(embeddings)):
                synset = synsets[i]
                synset_name2embeddings[synset.name()] = embeddings[i]

        self.__serialize_synset_name_to_embedding(synset_name2embeddings)
        return synset_name2embeddings

    def _get_cache_file_name(self):
        return 'serialized/' + self._cache_name

    def __serialize_synset_name_to_embedding(self, synset_name_to_embedding):
        print('Going to serialize synsets embeddings...')
        with open(self._get_cache_file_name(), 'wb') as file:
            pickle.dump(synset_name_to_embedding, file)

    def __deserialize_synset_name_to_embedding(self):
        with open(self._get_cache_file_name(), 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def __get_k_nearest_synsets(embedding, synsets, synset_name2embeddings, k):
        score_and_synsets = zip(map(
            lambda synset: cos_sim(embedding, synset_name2embeddings[synset.name()]),
            synsets
        ), synsets)

        return list(map(lambda x: x[1], sorted(score_and_synsets, key=lambda x: x[0], reverse=True)))[:k]

    @staticmethod
    def __get_appropriate_pos(pos: str):
        """
        TODO dirty hack
        """
        return 'n' if pos == 'noun' else 'v'

    @staticmethod
    def __get_synsets_filtered_by_pos(words, pos):
        synsets = map(wordnet.synsets, words)
        return list(filter(lambda x: x.pos() == pos, [item for synset in synsets for item in synset]))

    @staticmethod
    def _get_extended_synset_list(synsets):
        used_synsets = SetM()
        return list(
            filter(
                lambda s: len(s) > 0,
                map(
                    lambda s: list(
                        filter(
                            lambda t: used_synsets.add(t.name()),
                            [elem for ss in [wordnet.synsets(l) for l in s.lemma_names()] for elem in ss]
                        )
                    ),
                    synsets
                )
            )
        )

    @staticmethod
    def __get_synsets(words, pos):
        synsets = list(
            filter(
                lambda s: len(s) > 0,
                map(wordnet.synsets, words)
            )
        )
        filtered = list(filter(lambda x: x.pos() == pos, [item for synset in synsets for item in synset]))
        if len(filtered) > 0:
            return filtered
        synsets_flat = [item for synset in synsets for item in synset]
        synsets_extended = SentenceEmbeddingsBasedProcessor._get_extended_synset_list(synsets_flat)
        filtered = list(filter(lambda x: x.pos() == pos, [item for synset in synsets_extended for item in synset]))
        return filtered


class RelatedSynsetsSentenceEmbeddingsBasedProcessor(SentenceEmbeddingsBasedProcessor):
    def __init__(self, cache_name, use_cache=True):
        super().__init__(cache_name, use_cache)

    def _construct_candidates_definition_sentences(self, synsets: [Synset]):
        synsets_definition_extract_strategy = lambda synsets: list(
            map(
                lambda synset: ' '.join(map(lambda lemma: lemma.name(), synset.lemmas())),
                synsets
            )
        )
        result = list(
            map(
                lambda synset: ' '.join(synsets_definition_extract_strategy([synset])),
                synsets
            )
        )
        return result

class RelatedSynsetsConjunctureSentenceEmbeddingsBasedProcessor(SentenceEmbeddingsBasedProcessor):
    def __init__(self, cache_name, use_cache=True):
        super().__init__(cache_name, use_cache)

    def _construct_candidates_definition_sentences(self, synsets: [Synset]):
        synsets_definition_extract_strategy = lambda synsets: list(
            map(
                lambda synset: ' '.join(map(lambda lemma: lemma.name(), synset.lemmas())),
                synsets
            )
        )
        result = list(
            map(
                lambda synset: ' '.join(
                    synsets_definition_extract_strategy(synset.hypernyms()) + synsets_definition_extract_strategy(synset.hyponyms())
                ),
                synsets
            )
        )
        return result


class RelatedSynsetsFullConjunctureSentenceEmbeddingsBasedProcessor(SentenceEmbeddingsBasedProcessor):
    def __init__(self, cache_name, use_cache=True):
        super().__init__(cache_name, use_cache)

    def _construct_candidates_definition_sentences(self, synsets: [Synset]):
        synsets_definition_extract_strategy = lambda synsets: list(
            map(
                lambda synset: ' '.join(map(lambda lemma: lemma.name(), synset.lemmas())),
                synsets
            )
        )
        result = list(
            map(
                lambda synset: ' '.join(
                    synsets_definition_extract_strategy(synset.hypernyms()) + synsets_definition_extract_strategy(
                        synset.hyponyms()) + synsets_definition_extract_strategy([synset])
                ),
                synsets
            )
        )
        return result