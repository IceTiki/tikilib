import matplotlib.pyplot as plt
import numpy as np


def gene_fig_ax():
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot()
    ax.set_aspect(1)
    return fig, ax


def gene_gird(x_range=(-10, 10, 100), y_range=(-10, 10, 100), func=lambda x, y: x + y):
    x = np.linspace(*x_range)
    y = np.linspace(*y_range)
    X, Y = np.meshgrid(x, y)
    F = func(X, Y)
    return X, Y, F


def chinese_font_support() -> None:
    """matplotlib的中文显示支持的自动设置函数"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
