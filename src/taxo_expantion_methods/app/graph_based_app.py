from src.taxo_expantion_methods.dao.dao_factory import DaoFactory
from src.taxo_expantion_methods.engines.essential_words_aware_processor import EmbeddingsBasedEssentialWordsAwareProcessor
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddDataParser
from src.taxo_expantion_methods.format.result_printer import AllStatisticsResultPrinter


def run_poincare():
    dao_factory = DaoFactory()
    fasttext_dao = dao_factory.create_word_embeddings_dao()
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')

    poincare_dao = DaoFactory.create_poincare_word_embeddings_dao()
    processor = EmbeddingsBasedEssentialWordsAwareProcessor(poincare_dao)

    result = processor.process(words_to_add_data)
    # SemEvalTask2016FormatResultPrinter('result/poincare_result_2.csv').print_result(result)
    AllStatisticsResultPrinter('result/all_statistics_poincare.csv').print_result(result)


def run_node2vec():
    words_to_add_data = WordToAddDataParser.from_pandas('data/training_data.csv', '$')

    dao = DaoFactory.create_node_embeddings_dao()
    processor = EmbeddingsBasedEssentialWordsAwareProcessor(dao)

    result = processor.process(words_to_add_data)
    # SemEvalTask2016FormatResultPrinter('result/fasttext.csv').print_result(result)
    AllStatisticsResultPrinter('result/all_statistics_node.csv').print_result(result)
