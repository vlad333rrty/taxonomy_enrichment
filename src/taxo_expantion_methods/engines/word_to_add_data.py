import pandas as pd


class WordToAddData:
    def __init__(self, value: str, definition: str, pos: str, num: str):
        self.value = value
        self.definition = definition
        self.pos = pos
        self.num = num


class WordToAddDataParser:
    @staticmethod
    def from_pandas(path: str, sep: str) -> [WordToAddData]:
        result = []
        words_to_add = pd.read_csv(path, sep=sep)
        for index, row in words_to_add.iterrows():
            word = row['word']
            definition = row['definition']
            pos = row['pos']
            num = row['num']
            result.append(WordToAddData(word, definition, pos, num))
        return result
