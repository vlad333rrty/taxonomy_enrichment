from dao.dao_factory import DaoFactory
from engines.essential_words_aware_processor import EmbeddingsBasedEssentialWordsAwareProcessor
from engines.sentence_embeddings_based_processor import RelatedSynsetsFullConjunctureSentenceEmbeddingsBasedProcessor
from engines.word_to_add_data import WordToAddDataParser
from format.result_printer import AllStatisticsResultPrinter


def compare():
    processor = RelatedSynsetsFullConjunctureSentenceEmbeddingsBasedProcessor('synset_name_to_embedding_4')
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')
    labse_res = processor.process(words_to_add_data, 'LaBSE')

    dao = DaoFactory.create_node_embeddings_dao()
    processor = EmbeddingsBasedEssentialWordsAwareProcessor(dao)
    node_result = processor.process(words_to_add_data)

    dao = DaoFactory.create_poincare_word_embeddings_dao()
    processor = EmbeddingsBasedEssentialWordsAwareProcessor(dao)
    poincare_result = processor.process(words_to_add_data)

    AllStatisticsResultPrinter('result/labse_top5.csv').print_result(labse_res)
    AllStatisticsResultPrinter('result/poincare_top5').print_result(poincare_result)
    AllStatisticsResultPrinter('result/node_top5.csv').print_result(node_result)


compare()