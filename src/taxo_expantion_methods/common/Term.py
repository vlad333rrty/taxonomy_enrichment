class Term:
    def __init__(self, value, definition, pos=None):
        self.value = value
        self.definition = definition
        self.pos = pos

    def definition(self):
        return self.definition

    def value(self):
        return self.value

    def part_of_speech(self):
        return self.pos

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
