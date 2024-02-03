from matplotlib import pyplot as plt

from src.taxo_expantion_methods.TEMP.plot_monitor import PlotMonitor, Metric

pm = PlotMonitor()
for i in range(10):
    pm.accept(Metric('x', i))
    pm.plot()
plt.show()
