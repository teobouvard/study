f_daq_calls = 0


def f_daq(n):
    global f_daq_calls
    f_daq_calls += 1
    if n < 3:
        return n
    return f_daq(n - 1) * f_daq(n - 2) + (n - 3) * f_daq(n - 3)


if __name__ == "__main__":
    n = 20
    res = f_daq(n)
    # print(res)
    print(f_daq_calls)
