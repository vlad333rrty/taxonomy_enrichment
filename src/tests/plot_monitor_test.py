import random
import unittest

from matplotlib import pyplot as plt

from src.taxo_expantion_methods.common.plot_monitor import PlotMonitor, Metric


class PlotMonitorTest(unittest.TestCase):
    def test_1(self):
        plot_monitor = PlotMonitor()
        r = random.Random()
        xs = []
        for i in range(20):
            k = r.random() * (100 - 0.25 * i)
            plot_monitor.accept(Metric('test', k))
            plot_monitor.accept(Metric('test1', k / 100))
            xs.append(k)
        plot_monitor.plot()

        plt.show()


