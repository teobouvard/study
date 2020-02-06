import numpy as np
from matplotlib import pyplot as plt

def problem_2():
    x = np.arange(-10, 40)
    y1 = (30*60 - 12*x)/25
    y2 = (5*x)/2

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Constraint 1')
    ax.plot(x, y2, label='Constraint 2')
    

    constraints = x >= 0
    constraints &= y1>y2
    ax.fill_between(x, y1, y2, where=constraints, color='grey', alpha=0.5)

    intersections = np.array([(0, 0), (0, 72), (3600/149, 9000/149)])
    ax.scatter(intersections.T[0], intersections.T[1], c='red', marker='o', zorder=5)
    ax.annotate(r'$I_0$', xy=(0.7, -8))
    ax.annotate(r'$I_1$', xy=(-2, 75))
    ax.annotate(r'$I_2$', xy=(23, 65))

    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    ax.legend()
    ax.set(xlabel='x_1', ylabel='x_2')
    fig.savefig('graph1.png')
    plt.show()

def problem_3():
    x = np.arange(-5, 20)
    y1 = (300 - 30*x)/15
    y2 = (9000 - 500*x)/750
    y3 = 16 - x

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Constraint 1')
    ax.plot(x, y2, label='Constraint 2')
    ax.plot(x, y3, label='Constraint 3')
    

    intersections = np.array([(4, 12), (6, 8), (12, 4)])
    ax.scatter(intersections.T[0], intersections.T[1], c='red', marker='o', zorder=5)
    ax.fill(intersections.T[0], intersections.T[1], alpha=0.5, color='gray')
    ax.annotate(r'$I_1$', xy=(4, 14))
    ax.annotate(r'$I_2$', xy=(6, 4.5))
    ax.annotate(r'$I_3$', xy=(12, 6))

    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    ax.legend()
    ax.set(xlabel='x_1', ylabel='x_2')
    fig.savefig('graph2.png')
    plt.show()

def problem_4():
    x = np.arange(-5, 40, step=0.01)
    y1 = (50 - x)/2
    y2 = (240 - 8*x)/3

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Constraint 1')
    ax.plot(x, y2, label='Constraint 2')
    
    intersections = np.array([(0, 0), (30, 0), (330/13, 160/13), (0, 25)])
    ax.scatter(intersections.T[0], intersections.T[1], c='red', marker='o', zorder=5)
    ax.fill(intersections.T[0], intersections.T[1], alpha=0.5, color='gray')
    ax.annotate(r'$I_1$', xy=(-2, -7))
    ax.annotate(r'$I_2$', xy=(30, -8))
    ax.annotate(r'$I_3$', xy=(1, 28))
    ax.annotate(r'$I_4$', xy=(25, 16))

    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    ax.legend()
    ax.set(xlabel='x_1', ylabel='x_2')
    fig.savefig('graph3.png')
    plt.show()

if __name__ == '__main__':
    #problem_2()
    #problem_3()
    problem_4()