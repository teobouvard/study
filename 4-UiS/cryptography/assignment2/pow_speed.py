import timeit

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
    pow_times.append(timeit.timeit('pow({}, {}, {})'.format(a, x, m)))
    x *= 10

print(pow_times, raw_times)