from app.ClosestSynsetsEmbeddingsAggregator import PoincareClosestSynsetsEmbeddingsAggregator
from dao.dao_factory import DaoFactory
from engines.first_word_first_sense_processor import gather_first_synsets
from engines.sentence_embeddings_based_processor import SentenceEmbeddingsBasedProcessor, \
    RelatedSynsetsSentenceEmbeddingsBasedProcessor, RelatedSynsetsConjunctureSentenceEmbeddingsBasedProcessor, \
    RelatedSynsetsFullConjunctureSentenceEmbeddingsBasedProcessor
from engines.word_to_add_data import WordToAddDataParser
from result_printer import write_result, write_data_for_doc, SemEvalTask2016FormatResultPrinter


def sentence_based_LaBSE():
    processor = SentenceEmbeddingsBasedProcessor()
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    res = processor.process(words_to_add_data, 'LaBSE')
    SemEvalTask2016FormatResultPrinter('result/result2.csv').print_result(res)


def related_synsets_sentence_based_LaBSE():
    processor = RelatedSynsetsSentenceEmbeddingsBasedProcessor('synset_name_to_embedding_2')
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    res = processor.process(words_to_add_data, 'LaBSE')
    SemEvalTask2016FormatResultPrinter('result/result3.csv').print_result(res)


def related_synsets_sentence_based_LaBSE_hypernyms_and_hyponyms():
    processor = RelatedSynsetsConjunctureSentenceEmbeddingsBasedProcessor('synset_name_to_embedding_3')
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    res = processor.process(words_to_add_data, 'LaBSE')
    SemEvalTask2016FormatResultPrinter('result/result4.csv').print_result(res)

def related_synsets_sentence_based_LaBSE_hypernyms_and_hyponyms_and_word():
    processor = RelatedSynsetsFullConjunctureSentenceEmbeddingsBasedProcessor('synset_name_to_embedding_4', use_cache=False)
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    res = processor.process(words_to_add_data, 'LaBSE')
    SemEvalTask2016FormatResultPrinter('result/result5.csv').print_result(res)

related_synsets_sentence_based_LaBSE_hypernyms_and_hyponyms_and_word()