class WordNetSubgraphProvider:
    def __init__(self, all_synsets):
        self.__all_synsets = all_synsets

    def get_food_synsets(self):
        res = []
        for s in self.__all_synsets:
            if 'food' in s.lexname():
                res.append(s)
        return res