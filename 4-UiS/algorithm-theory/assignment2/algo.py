def largest_set(x):
    maxlen = 0
    maxseq = 0
    for i in range(len(x) - 1):
        j = i + 1
        n = 2
        while x[j] - x[i] <= 1:
            j += 1
            n += 1
        if n - 1 > maxlen:
            maxlen = n - 1
            maxseq = i
    return maxseq, maxlen


if __name__ == "__main__":
    x = [1, 1.25, 1.25, 1.5, 2, 2.5, 2.75, 3, 4, 5, 7.5, 10]
    i, m = largest_set(x)
    print(i, m)
