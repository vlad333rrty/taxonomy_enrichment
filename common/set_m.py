class SetM:
    def __init__(self):
        self.__collection = set()

    def add(self, elem):
        is_present = elem in self.__collection
        if is_present:
            return False
        self.__collection.add(elem)
        return True