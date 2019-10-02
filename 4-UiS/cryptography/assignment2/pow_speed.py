from matplotlib import pyplot as plt
import timeit

LIMIT = 10e5
x = 1
a = 2
m = 2

raw_times = []
pow_times = []
x_list = []

while x < LIMIT:
    x_list.append(x)
    raw_times.append(timeit.timeit('{} ** {} % {}'.format(a, x, m), number=1))
    pow_times.append(timeit.timeit('pow({}, {}, {})'.format(a, x, m), number=1))
    x *= 10

    plt.plot(x_list, raw_times)
    plt.plot(x_list, pow_times)
    plt.show()
