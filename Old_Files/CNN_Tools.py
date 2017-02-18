import matplotlib
import numpy as np

from matplotlib import pyplot as plt


def plot_filters(layer, x, y):
    """Plot the filters for net after the (convolutional) layer number layer.
    They are plotted in x by y format. So, for example, if we
    have 20 filters after layer 0, then we can call plot_filters(l_conv1, 5 , 4)
    to get  a 5 by 4 plot of all filters.
    """

    filters = layer.W.get_value()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j+1)
        ax.matshow(filters[j][0], cmap=plt.cm.gray)

        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
