# __author__ = 'dimitrios'
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(x, y, name=None, xlabel=None, ylabel=None):
    f, ax = plt.subplots(figsize=(15, 15))

    diff = x - y
    m = np.where(diff > 0)
    up_x = x[m]
    up_y = y[m]
    red_str = 'Red are %d' % len(m[0])
    m = np.where(diff <= 0)
    blue_str = 'Blue are %d' % len(m[0])
    str = red_str + '\n' + blue_str
    print str

    low_x = x[m]
    low_y = y[m]
    ax.set_xlim([-11, 0])
    ax.set_ylim([-11, 0])
    ax.scatter(low_x, low_y, marker='.', edgecolors=None)
    ax.scatter(up_x, up_y, c='r', marker='.', edgecolors=None)
    ax.text(0, 0, str)
    # ax.scatter(x, y)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
