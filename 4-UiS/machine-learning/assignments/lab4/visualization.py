import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = ["red", "blue", "green", "black"]


def plot_2D(train, test):
    _, ax = plt.subplots(1, 2)
    for i, data in enumerate([train, test]):
        for c, class_ in enumerate(data):
            ax[i].scatter(*class_, c=colors[c], label=f"class {c+1}")
    plt.legend()


def plot_3D(train, test):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    for i, data in enumerate([train, test]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for c, class_ in enumerate(data):
            ax.scatter(*class_, c=colors[c], label=f"class {c+1}")
    plt.legend()


if __name__ == "__main__":
    _, _, _, _, X_3D3cl_ms, _, _, _, _, Y_3D3cl_ms = np.load(
        "data/lab4_2.p", allow_pickle=True
    )
    plot_3D(X_3D3cl_ms, Y_3D3cl_ms)

