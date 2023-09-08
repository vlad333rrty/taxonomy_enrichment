from abc import ABC, abstractmethod

from nltk.corpus.reader import Synset

from engines.processor_base import ProcessingResult


def write_result(data_list, dest_path):
    """
    format: withdef.1	primer#n#1	attach - separator == \t
    :return:
    """
    contents = ''
    for x in data_list:
        contents += '{}\t{}\t{}\n'.format(x[0], '' if x[1] is None else x[1].name(), x[2])
    with open(dest_path, 'w') as file:
        file.write(contents)


def write_data_for_doc(data_list, path):
    """
    format word to add, first word from gloss, sysnsets of this word
    """
    contents = ''
    for data in data_list:
        contents += '{}, {}, {}\n'.format(data[0], data[1], data[2])
    with open(path, 'w') as file:
        file.write(contents)


class ResultPrinter(ABC):
    @abstractmethod
    def print_result(self, results: [ProcessingResult]):
        raise NotImplementedError()


class SemEvalTask2016FormatResultPrinter(ResultPrinter):
    def __init__(self, dest_path):
        self.__dest_path = dest_path

    def print_result(self, results: [ProcessingResult]):
        contents = ''
        for x in results:
            etalon_synset = x.candidate_synsets[0]
            contents += '{}\t{}\t{}\n'.format(x.word_token, etalon_synset.name(), x.strategy)
        with open(self.__dest_path, 'w') as file:
            file.write(contents)


class AllStatisticsResultPrinter(ResultPrinter):
    def __init__(self, dest_path):
        self.__dest_path = dest_path

    def print_result(self, results: [ProcessingResult]):
        contents = ''
        for x in results:
            synsets_str = ', '.join(map(Synset.name, x.candidate_synsets))
            contents += '{}\t{}\t{}\n'.format(x.word_token, synsets_str, x.strategy)
        with open(self.__dest_path, 'w') as file:
            file.write(contents)


