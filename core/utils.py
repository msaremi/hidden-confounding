import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def _set_axis_properties(**kwargs):
    plt.yscale(kwargs['yscale'] if 'yscale' in kwargs else 'linear')
    plt.xscale(kwargs['xscale'] if 'xscale' in kwargs else 'linear')
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else '')
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else '')
    plt.xlim(kwargs['xlim'] if 'xlim' in kwargs else None)
    plt.ylim(kwargs['ylim'] if 'ylim' in kwargs else None)


def draw_contourf(X, Y, Z, **kwargs):
    matplotlib.rc('font', size=15)
    fig = plt.figure()
    cmap = matplotlib.colormaps['YlGnBu']
    plt.contourf(X, Y, Z, cmap=cmap, levels=7)
    _set_axis_properties(**kwargs)
    norm = matplotlib.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca())
    cbar.set_label(kwargs['cbar_label'])
    return fig


def draw_plots(X, Y_min, Y_med, Y_max, names, **kwargs):
    matplotlib.rc('font', size=15)
    fig = plt.figure()

    for i in np.arange(Y_med.shape[0]):
        plt.plot(X, Y_med[i, :], label=names[i], marker='o')
        plt.fill_between(X, Y_min[i, :], Y_max[i, :], alpha=0.5)

    _set_axis_properties(**kwargs)
    plt.gca().yaxis.set_label_position("right")
    plt.legend()
    plt.grid()
    return fig


def save_fig(file_name, figure):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(file_name)