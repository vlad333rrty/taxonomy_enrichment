import sys
from datetime import datetime


class ConsoleLogger:
    def __init__(self, parent):
        self.__parent = parent

    def info(self, message, *args):
        now = datetime.now()
        print('INFO: {}, {}: {}'.format(self.__parent, now, message).format(*args))

    def warn(self, message, *args):
        now = datetime.now()
        print('WARN: {}, {}: {}'.format(self.__parent, now, message).format(*args))

    def error(self, message, *args):
        now = datetime.now()
        print('ERROR: {}, {}: {}'.format(self.__parent, now, message).format(*args), file=sys.stderr)
