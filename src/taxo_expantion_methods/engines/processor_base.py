from abc import ABC, abstractmethod

from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddData

class ProcessingResult:
    def __init__(self, word, word_token, candidate_synsets, strategy):
        self.word = word
        self.word_token = word_token
        self.candidate_synsets = candidate_synsets
        self.strategy = strategy

class Processor(ABC):
    @abstractmethod
    def process(self, words_to_add_data: [WordToAddData]) -> [ProcessingResult]:
        raise NotImplementedError()



