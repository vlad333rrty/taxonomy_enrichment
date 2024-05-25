class Term:
    def __init__(self, value, definition, pos=None):
        self.__value = value
        self.__definition = definition
        self.__pos = pos

    def definition(self):
        return self.__definition

    def value(self):
        return self.__value

    def part_of_speech(self):
        return self.__pos

    def __str__(self):
        return self.__value

    def __repr__(self):
        return self.__value
