import numpy as np


def matrix_chain_order(p):
    n = len(p) - 1
    m = np.zeros((n, n), dtype=int)
    s = np.zeros((n, n), dtype=int)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i][j] = np.iinfo(int).max
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k

    return m, s


def inverse_matrix_chain_order(p):
    n = len(p) - 1
    m = np.zeros((n, n), dtype=int)
    s = np.zeros((n, n), dtype=int)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q > m[i, j]:
                    m[i, j] = q
                    s[i, j] = k

    return m, s


def print_optimal_parens(s, i, j):
    if i == j:
        print(f"A_{i+1}", end="")
    else:
        print("(", end="")
        print_optimal_parens(s, i, s[i, j])
        print_optimal_parens(s, s[i, j] + 1, j)
        print(")", end="")


if __name__ == "__main__":
    p = [30, 35, 15, 5, 10, 20, 25]

    m, s = matrix_chain_order(p)
    print(m)
    print(s)
    print(f"Optimal cost : {m[0, -1]}")
    print_optimal_parens(s, 0, len(s) - 1)
    print()

    p = [30, 35, 15, 5, 10, 25]
    m, s = inverse_matrix_chain_order(p)
    print(m)
    print(s)
    print(f"Optimal cost : {m[0, -1]}")
    print_optimal_parens(s, 0, len(s) - 1)
    print()
