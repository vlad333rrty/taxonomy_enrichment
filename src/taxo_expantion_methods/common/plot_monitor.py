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
        self.__lock = threading.Lock()

    def accept(self, metric: Metric):
        self.__lock.acquire()

        if metric.label not in self.__buffers:
            self.__buffers[metric.label] = []
        self.__buffers[metric.label].append(metric.value)

        self.__lock.release()

    def plot(self):
        self.__lock.acquire()
        plt.ion()
        display.clear_output(wait=False)
        plt.show()

        for label in self.__buffers:
            plt.plot(self.__buffers[label], label=label)

        plt.legend()
        plt.show()
        plt.ioff()
        self.__lock.release()


    def clear(self):
        self.__lock.acquire()
        self.__buffers.clear()
        self.__lock.release()
