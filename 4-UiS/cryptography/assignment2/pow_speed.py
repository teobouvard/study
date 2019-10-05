import os
import timeit

from matplotlib import pyplot as plt

LIMIT = 10e10
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
    x *=2

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log')

plt.plot(x_list, raw_times, marker='o')
plt.plot(x_list, pow_times, marker='o')
plt.xlabel('i, log scale')
plt.xticks()
plt.ylabel('computation time (s)')
plt.legend(['y = x ** i mod M', 'y = pow(x, i, M)'])
os.makedirs('img', exist_ok=True)
plt.savefig('img/pow_speed.png', bbox_inches='tight')
plt.show()
