import numpy as np

def norm1D(mu, sigma, x):
    n, d = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-1 / 2 * np.square((x[i] - mu)) / (np.square(sigma)))
    return p


def norm2D(mu, sigma, x1, x2):
    mesh = np.meshgrid(x1, x2, indexing='ij')

    # precompute constant value and initialize result array
    p = np.zeros([len(x1), len(x2)])
    k = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    sigma_inv = np.linalg.inv(sigma)

    for i, u in enumerate(x1):
        for j, v in enumerate(x2):
            x = np.array([u, v]).reshape(-1, 1)
            M = (x-mu).T @ sigma_inv @ (x-mu)
            p[i][j] = k * np.exp(-0.5 * M)

    return p, mesh
    