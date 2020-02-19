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
    learning_rate = 0.5
    theta_i = 1
    theta = [1, 1, 1]
    threshold = 1
    y = [1, 1, 2, 2]
    N = len(y)
    i = 0
    while learning_rate*() > threshold:
        i = (i+1)%N
        learning_rate = learning_rate/i
        theta = theta 


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


def setup():
    omega_1 = np.array([[-3, -2], [1, -1]])
    omega_2 = np.array([[-4, -1]])

    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.scatter(*omega_1.T, marker='x', color='k', label='$\omega_1$')
    ax.scatter(*omega_2.T, marker='o', edgecolors='k', facecolors='none', label='$\omega_2$')

    ax.set(xlabel='$x_1$', ylabel='$x_2$', aspect='equal', xlim=(-5, 2), ylim=(-4, 2))
    ax.grid(which='both')

    x = np.arange(-10, 10)
    return x, fig, ax


def problem_5():
    x, fig, ax = setup()

    y = 2/3 * x
    line = ax.plot(x, y, 'k--', label='$\\theta^T x_1 = 0$')
    ax.legend()
    fig.savefig('graph5.png')

    ax.fill_between(x, y, -10, color='red', alpha=0.2)
    ax.fill_between(x, y, 10, color='blue', alpha=0.2)
    fig.savefig('graph6.png')


def problem_6():
    x, fig, ax = setup()
    y = -x
    line = ax.plot(x, y, 'k--', label='$\\theta^T x_2 = 0$')
    ax.legend()
    fig.savefig('graph7.png')

    ax.fill_between(x, y, -10, color='red', alpha=0.2)
    ax.fill_between(x, y, 10, color='blue', alpha=0.2)
    fig.savefig('graph8.png')


def problem_7():
    x, fig, ax = setup()
    y = 1/4 * x
    line = ax.plot(x, y, 'k--', label='$\\theta^T x_3 = 0$')
    ax.legend()
    fig.savefig('graph9.png')

    ax.fill_between(x, y, -10, color='red', alpha=0.2)
    ax.fill_between(x, y, 10, color='blue', alpha=0.2)
    fig.savefig('graph10.png')


def batch_perceptron():
    samples = np.array([[-3, -2], [1, -1], [-4, -1]])
    labels = np.array([1, 1, -1]).T

    theta = 0
    learning_rate = 1
    criterion = 0
    converged = False

    while not converged:
        step = labels @ samples
        theta += learning_rate * step
        converged = np.linalg.norm(step) <= criterion

    print(f'theta : {theta}')


def batch_perceptron_adaptative():
    samples = np.array([[-3, -2], [1, -1], [-4, -1]])
    labels = np.array([1, 1, -1]).T

    theta = 0
    learning_rate_init = 1
    i = 0
    criterion = 0
    converged = False

    while not converged:
        i += 1
        learning_rate = learning_rate_init / i
        step = labels @ samples
        theta += learning_rate * step
        converged = np.linalg.norm(step) <= criterion

    print(f'theta : {theta}')


if __name__ == '__main__':
    # problem_1()
    # problem_2()
    # problem_3()

    # problem_5()
    # problem_6()
    # problem_7()

    # batch_perceptron()
    batch_perceptron_adaptative()
