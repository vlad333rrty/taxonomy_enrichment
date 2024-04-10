from ruwordnet import ruwordnet


class RuWordnetDao:
    @staticmethod
    def get_ru_wn_20():
        return ruwordnet.RuWordNet('data/wordnets/ruwordnet.db')

    @staticmethod
    def get_ru_wn_21():
        return ruwordnet.RuWordNet('data/wordnets/ruwordnet-2021.db')