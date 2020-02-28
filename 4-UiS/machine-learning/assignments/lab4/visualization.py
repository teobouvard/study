import numpy as np
from matplotlib import pyplot as plt

colors = ["red", "blue", "green", "black"]


def plot_2D(train, test):
    fig, ax = plt.subplots(1, 2)
    for i, data in enumerate([train, test]):
        for c, class_ in enumerate(data):
            ax[i].scatter(*class_, c=colors[c], label=f"class {c+1}")
    plt.legend()


if __name__ == "__main__":
    X_2D3cl, _, _, _, _, Y_2D3cl, _, _, _, _ = np.load(
        "lab4/data/lab4_2.p", allow_pickle=True
    )
    plot_2D(X_2D3cl, Y_2D3cl)

