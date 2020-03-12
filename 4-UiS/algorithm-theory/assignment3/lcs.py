import numpy as np


def lcs(X, Y):
    m = len(X) + 1
    n = len(Y) + 1
    b = np.zeros((m, n), dtype=object)
    c = np.zeros((m, n), dtype=int)
    for i in range(1, m):
        for j in range(1, n):
            if X[i - 1] == Y[j - 1]:
                b[i, j] = "NW"
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i - 1, j] >= c[i, j - 1]:
                b[i, j] = "UP"
                c[i, j] = c[i - 1, j]
            else:
                b[i, j] = "LE"
                c[i, j] = c[i, j - 1]
    return b, c


def print_lcs(b, X, i, j, res):
    if (i == 0) or (j == 0):
        return ""
    if b[i, j] == "NW":
        res += print_lcs(b, X, i - 1, j - 1, res) + X[i - 1]
    elif b[i, j] == "UP":
        res += print_lcs(b, X, i - 1, j, res)
    else:
        res += print_lcs(b, X, i, j - 1, res)
    return res


if __name__ == "__main__":
    Xl = ["ABCBDAB", "PRESIDENT", "ALGORITHM"]
    Yl = ["BDCABA", "PROVIDENCE", "ALIGNMENT"]

    for X, Y in zip(Xl, Yl):
        b, c = lcs(X, Y)
        print(f"LCS of {X} and {Y} : {print_lcs(b, X, len(X), len(Y), '')}")

    X = "CACAQ"
    Y = "CADACA"

    b, c = lcs(X, Y)
    print(f"LCS of {X} and {Y} : {print_lcs(b, X, len(X), len(Y), '')}")
    print(c)
    print(b)
