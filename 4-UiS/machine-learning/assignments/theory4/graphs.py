import numpy as np
from matplotlib import pyplot as plt


def problem_1():
    omega_1 = np.array([[1, 2], [2, 0]])
    omega_2 = np.array([[3, 1], [2, 3]])

    fig, ax = plt.subplots()
    ax.scatter(*omega_1.T, marker='x', label='$\omega_1$')
    ax.scatter(*omega_2.T, marker='o', edgecolors='r',
               facecolors='none', label='$\omega_2$')

    ax.legend()
    ax.set(xlabel='$x_1$', ylabel='$x_2$',
           aspect='equal', xlim=(-1, 4), ylim=(-1, 4))
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
    init_learning_rate = 0.5
    theta = np.array([1., 1., 1.]).T
    threshold = 1
    y = np.array([1, 1, -1, -1]).T
    x = np.array([[1, 2, 1], [2, 0, 1], [3, 1, 1], [2, 3, 1]])
    N = len(y)
    i = 0
    has_converged = False

    while not has_converged:
        has_converged = True
        for sample, target in zip(x, y):
            i += 1
            learning_rate = init_learning_rate / i
            step = learning_rate*(target - theta.T * target @
                                  sample)*(target * sample)

            if np.linalg.norm(step) > threshold:
                has_converged = False
                theta += step

            print(f'Step : {step}')
            print(f'Step size : {np.linalg.norm(step)}')
            print(f'Theta : {theta}')
            print()

    print(f'n iter : {i}')


def problem_3():
    omega_1 = np.array([[1, 2], [2, 0]])
    omega_2 = np.array([[3, 1], [2, 3]])

    fig, ax = plt.subplots()
    ax.scatter(*omega_1.T, marker='x', label='$\omega_1$')
    ax.scatter(*omega_2.T, marker='o', edgecolors='r',
               facecolors='none', label='$\omega_2$')

    ax.legend()
    ax.set(xlabel='$x_1$', ylabel='$x_2$',
           aspect='equal', xlim=(-1, 4), ylim=(-1, 4))
    ax.grid(which='both')

    x = np.arange(-1, 5)
    y = (1/0.42) * (0.79 * x + 0.32)
    ax.plot(x, y, 'k-', label='$\theta^{(2)}$')
    fig.savefig('graph4.png')


def setup():
    omega_1 = np.array([[-3, -2], [1, -1]])
    omega_2 = np.array([[4, 1]])

    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.scatter(*omega_1.T, marker='x', color='k', label='$\omega_1$', zorder=5)
    ax.scatter(*omega_2.T, marker='o', edgecolors='k',
               facecolors='none', label='$\omega_2$', zorder=5)

    ax.set(xlabel='$x_1$', ylabel='$x_2$',
           aspect='equal', xlim=(-4, 5), ylim=(-5, 4))
    ax.grid(which='both')

    x = np.arange(-10, 10)
    return x, fig, ax


def problem_5():
    x, fig, ax = setup()

    y = -3/2 * x
    ax.plot(x, y, 'k--', label='$\\theta^T x_1 = 0$')
    ax.legend()
    fig.savefig('graph5.png')

    ax.fill_between(x, y, -10, color='red', alpha=0.3)
    ax.fill_between(x, y, 10, color='blue', alpha=0.3)
    fig.savefig('graph6.png')


def problem_6():
    x, fig, ax = setup()
    y = x
    line = ax.plot(x, y, 'k--', label='$\\theta^T x_2 = 0$')
    ax.legend()
    fig.savefig('graph7.png')

    ax.fill_between(x, y, -10, color='red', alpha=0.3)
    ax.fill_between(x, y, 10, color='blue', alpha=0.3)
    fig.savefig('graph8.png')


def problem_7():
    x, fig, ax = setup()
    y = -4 * x
    line = ax.plot(x, y, 'k--', label='$\\theta^T x_3 = 0$')
    ax.legend()
    fig.savefig('graph9.png')

    ax.fill_between(x, y, 10, color='red', alpha=0.3)
    ax.fill_between(x, y, -10, color='blue', alpha=0.3)
    fig.savefig('graph10.png')


def solution_region():
    x, fig, ax = setup()

    y = -3/2 * x
    ax.plot(x, y, 'k--', label='$\\theta^T x_1 = 0$')
    ax.fill_between(x, y, -10, color='red', alpha=0.3)
    ax.fill_between(x, y, 10, color='blue', alpha=0.3)

    y = x
    ax.plot(x, y, 'k--', label='$\\theta^T x_2 = 0$')
    ax.fill_between(x, y, -10, color='red', alpha=0.3)
    ax.fill_between(x, y, 10, color='blue', alpha=0.3)

    y = -4 * x
    line = ax.plot(x, y, 'k--', label='$\\theta^T x_3 = 0$')
    ax.legend()

    ax.fill_between(x, y, 10, color='red', alpha=0.3)
    ax.fill_between(x, y, -10, color='blue', alpha=0.3)

    fig.savefig('solution_region.png')


def batch_perceptron():
    samples = np.array([[-3, -2], [1, -1], [-4, -1]])
    labels = np.array([1, 1, -1]).T
    theta = np.array([0, 0]).T

    criterion = 0
    has_converged = False
    i = 0

    while not has_converged:
        i += 1
        misclassified = []
        has_converged = True
        for s, l in zip(samples, labels):
            if l*theta.T @ s <= criterion:
                misclassified.append((s, l))

        if misclassified:
            print(f'Step {i}')
            print(f'Missclassified : {misclassified}')
            has_converged = False
            change = np.sum([x*y for x, y in misclassified], axis=0)
            print(f'Change : {change}')
            theta += change
            print(f'New theta : {theta}')


def batch_perceptron_adaptative():
    samples = np.array([[-3, -2], [1, -1], [-4, -1]])
    labels = np.array([1, 1, -1]).T
    theta = np.array([0., 0.]).T

    learning_rate_init = 1
    i = 0
    criterion = 0.00000001
    has_converged = False

    while not has_converged:
        i += 1
        misclassified = []
        has_converged = True
        for s, l in zip(samples, labels):
            if l*theta.T @ s <= criterion:
                misclassified.append((s, l))

        if misclassified:
            print(f'Step {i}')
            print(f'Missclassified : {misclassified}')
            has_converged = False
            change = learning_rate_init/i * \
                np.sum([x*y for x, y in misclassified], axis=0)
            print(f'Change norm : {np.linalg.norm(change)}')
            print(f'Change : {change}')
            theta += change
            print(f'New theta : {theta}')


if __name__ == '__main__':
    # problem_1()
    # problem_2()
    # problem_3()
    # problem_5()
    # problem_6()
    # problem_7()
    # solution_region()
    # batch_perceptron()
    batch_perceptron_adaptative()
