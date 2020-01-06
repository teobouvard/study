import numpy as np

def norm1D(mu, sigma, x):
    n, d = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-1 / 2 * np.square((x[i] - mu)) / (np.square(sigma)))

    return p

def norm2D(mu, sigma, x):
    [n, d] = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-1 / 2 * np.square((x[i] - mu)) / (np.square(sigma)))

    return p
