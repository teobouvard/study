import numpy as np
from matplotlib import pyplot as plt


def problem_1():
    omega_1 = np.array([[1, 2], [2, 0]])
    omega_2 = np.array([[3, 1], [2, 3]])

    fig, ax = plt.subplots()
    ax.scatter(*omega_1.T, marker='x', label='$\omega_1$')
    ax.scatter(*omega_2.T, marker='o', edgecolors='r', facecolors='none', label='$\omega_2$')

    ax.legend()
    ax.set(xlabel='$x_1$', ylabel='$x_2$', aspect='equal', xlim=(-1, 4), ylim=(-1, 4))
    ax.grid(which='both')
    fig.savefig('graph1.png')

    x = np.arange(-1, 5)
    y = 11/2 - 2*x
    ax.plot(x, y, 'k-', label='g')


    ax.legend()
    fig.savefig('graph2.png')

    y_prime = 27/4 - (5/2)*x
    ax.plot(x, y_prime, 'k--', label='g\'')

    ax.legend()
    fig.savefig('graph3.png')


if __name__ == '__main__':
    problem_1()
