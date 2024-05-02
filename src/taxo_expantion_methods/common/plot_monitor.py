import threading

from matplotlib import pyplot as plt
from IPython import display


class Metric:
    def __init__(self, label, value):
        self.label = label
        self.value = value


class PlotMonitor:
    def __init__(self):
        self.__buffers = {}
        self.__sma_by_label = {}
        self.__lock = threading.Lock()

    def accept(self, metric: Metric):
        self.__lock.acquire()
        label = metric.label
        if label not in self.__buffers:
            self.__buffers[label] = [] # todo there is no need to store all these values
            self.__sma_by_label[label] = []
        self.__buffers[label].append(metric.value)
        sma = self.__get_sma(label)
        self.__sma_by_label[label].append(sma)

        self.__lock.release()

    def plot(self):
        self.__lock.acquire()
        plt.ion()
        display.clear_output(wait=False)
        plt.show()

        for label in self.__buffers:
            plt.plot(self.__sma_by_label[label], label=label)

        plt.legend()
        plt.show()
        plt.ioff()
        self.__lock.release()

    def __get_sma(self, label, n=10):
        accum = 0
        values = self.__buffers[label]
        bound = min(len(values), n)
        for i in range(0, bound):
            accum += values[-i]
        return accum / bound

    def clear(self):
        self.__lock.acquire()
        self.__buffers.clear()
        self.__sma_by_label.clear()
        self.__lock.release()
