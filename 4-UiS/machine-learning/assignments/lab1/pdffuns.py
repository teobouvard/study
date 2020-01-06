import numpy as np

def norm1D(mu, sigma, x):
    n, d = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-1 / 2 * np.square((x[i] - mu)) / (np.square(sigma)))

    return p

def norm2D(mu, sigma, mesh):
    n1, d1 = np.shape(mesh[0])
    n2, d2 = np.shape(mesh[1])

    # we make sure that the grid is a square
    assert n1 == n2
    assert d1 == d2

    # precompute constant values and initialize result
    p = np.zeros((n1, d1))
    k = 1 / (np.sqrt(2 * np.pi * np.linalg.det(sigma)))
    sigma_inv = np.linalg.inv(sigma)

    for i in np.arange(n1):
        for j in np.arange(d1):
            x = np.array([mesh[0][i][j], mesh[1][i][j]]).reshape(-1, 1)
            M = np.matmul(np.subtract(x, mu).T, sigma_inv)
            M = np.matmul(M, np.subtract(x,mu))
            p[i][j] = k * np.exp(-0.5 * M)

    return p

if __name__ == '__main__':
    x1 = np.arange(-10, 10.5, 0.5).reshape(-1, 1)
    x2 = np.arange(-9, 10.5, 0.5).reshape(-1, 1)
    grid = np.meshgrid(x1, x2)
    mu = np.array([1, 1]).reshape(-1, 1)
    covariance_matrix = np.array([5, 3, 3, 5]).reshape(2, 2)
    p = norm2D(mu, covariance_matrix, grid)