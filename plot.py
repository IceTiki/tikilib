import matplotlib.pyplot as plt
import numpy as np


def getFigAx():
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot()
    ax.set_aspect(1)
    return fig, ax


def getGird(xRange=(-10, 10, 100), yRange=(-10, 10, 100), func=lambda x, y: x+y):
    x = np.linspace(*xRange)
    y = np.linspace(*yRange)
    X, Y = np.meshgrid(x, y)
    F = func(X, Y)
    return X, Y, F
