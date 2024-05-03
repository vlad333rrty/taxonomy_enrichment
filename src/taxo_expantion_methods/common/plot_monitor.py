import threading
from collections import deque

from matplotlib import pyplot as plt
from IPython import display


class Metric:
    def __init__(self, label, value):
        self.label = label
        self.value = value


class PlotMonitor:
    """
    Calculates SMA (simple mean average) with given window size
    """
    def __init__(self, window_size=10):
        self.__window_size = window_size
        self.__label_to_value_queue = {}
        self.__sma_by_label = {}
        self.__lock = threading.Lock()

    def accept(self, metric: Metric):
        self.__lock.acquire()

        label = metric.label
        if label not in self.__label_to_value_queue:
            self.__label_to_value_queue[label] = deque()
            self.__sma_by_label[label] = []
        values = self.__label_to_value_queue[label]
        values.append(metric.value)
        if len(values) > self.__window_size + 1:
            values.popleft()
        sma_s = self.__sma_by_label[label]
        if len(values) == self.__window_size + 1:
            sma = self.__get_sma_dynamic(label, sma_s[-1])
            sma_s.append(sma)
        elif len(values) < self.__window_size:
            sma_s.append(metric.value)
        elif len(values) == self.__window_size:
            sma = self.__get_sma(label)
            sma_s.append(sma)

        self.__lock.release()

    def plot(self):
        self.__lock.acquire()
        plt.ion()
        display.clear_output(wait=False)
        plt.show()

        for label in self.__sma_by_label:
            sma_s = self.__sma_by_label[label]
            data = sma_s if len(sma_s) > 0 else self.__label_to_value_queue[label]
            plt.plot(data, label=label)

        plt.legend()
        plt.show()
        plt.ioff()
        self.__lock.release()

    def __get_sma(self, label):
        accum = 0
        values = self.__label_to_value_queue[label]
        bound = min(len(values), self.__window_size)
        for i in range(0, bound):
            accum += values[i]
        return accum / bound

    def __get_sma_dynamic(self, label, previous_value):
        values = self.__label_to_value_queue[label]
        return previous_value + (values[-1] - values[0]) / self.__window_size

    def __should_init_sma_buffer(self, sma_s, values):
        return len(sma_s) == 0 and len(values) == self.__window_size

    def clear(self):
        self.__lock.acquire()
        self.__label_to_value_queue.clear()
        self.__sma_by_label.clear()
        self.__lock.release()
