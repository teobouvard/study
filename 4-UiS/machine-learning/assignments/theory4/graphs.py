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


def problem_2():
    init_learning_rate = 1.
    theta = np.array([1., 1., 1.]).T
    threshold = 1
    y = np.array([-1, -1, 1, 1]).T
    x = np.array([[1, 2, 1], [2, 0, 1], [3, 1, 1], [2, 3, 1]])
    N = len(y)
    i = 0
    error = 100

    while error > threshold:
        i += 1
        sample_index = (i-1) % N
        learning_rate = init_learning_rate / i
        descent = learning_rate*(y[sample_index] - theta.T * y[sample_index] @
                                 x[sample_index])*(y[sample_index] * x[sample_index])
        theta += descent
        error = np.linalg.norm(descent)
        print(f'Descent : {descent}')
        print(f'Step : {error}')
        print(f'Theta : {theta}')
        print()


def problem_3():
    omega_1 = np.array([[1, 2], [2, 0]])
    omega_2 = np.array([[3, 1], [2, 3]])

    fig, ax = plt.subplots()
    ax.scatter(*omega_1.T, marker='x', label='$\omega_1$')
    ax.scatter(*omega_2.T, marker='o', edgecolors='r', facecolors='none', label='$\omega_2$')

    ax.legend()
    ax.set(xlabel='$x_1$', ylabel='$x_2$', aspect='equal', xlim=(-1, 4), ylim=(-1, 4))
    ax.grid(which='both')

    x = np.arange(-1, 5)
    y = 3/8 * x + 1/6
    ax.plot(x, y, 'k-', label='$\theta^{(2)}$')
    fig.savefig('graph4.png')


if __name__ == '__main__':
    # problem_1()
    # problem_2()
    problem_3()
