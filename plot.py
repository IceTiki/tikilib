import matplotlib.pyplot as plt
import numpy as np


def gene_fig_ax():
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot()
    ax.set_aspect(1)
    return fig, ax


def gene_gird(xRange=(-10, 10, 100), yRange=(-10, 10, 100), func=lambda x, y: x + y):
    x = np.linspace(*xRange)
    y = np.linspace(*yRange)
    X, Y = np.meshgrid(x, y)
    F = func(X, Y)
    return X, Y, F


def chinese_font_support():
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
