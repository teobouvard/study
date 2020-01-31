import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
from matplotlib import cm


def norm2D(mu, sigma, min_x1=-10, max_x1=10, min_x2=-10, max_x2=10):
    x1 = np.linspace(min_x1, max_x1, num=10*(max_x1-min_x1)).reshape(-1, 1)
    x2 = np.linspace(min_x2, max_x2, num=10*(max_x2-min_x2)).reshape(-1, 1)
    mesh = np.meshgrid(x1, x2, indexing='ij')

    # precompute constant value and initialize result array
    p = np.zeros([len(x1), len(x2)])
    k = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    sigma_inv = np.linalg.inv(sigma)

    for i, u in enumerate(x1):
        for j, v in enumerate(x2):
            x = np.array([u, v]).reshape(-1, 1)
            M = (x-mu).T @ sigma_inv @ (x-mu)
            p[i, j] = k * np.exp(-0.5 * M)

    return mesh[0], mesh[1], p


def parzen(samples, h1):
    cov = np.diag([h1**2, h1**2])
    x1, x2, p = norm2D(samples[0].reshape(-1, 1), cov)
    for s in samples[1:]:
        p += norm2D(s.reshape(-1, 1), cov)[2]
    return x1, x2, p / len(samples)


def knn(samples_1, samples_2, k, min_x1=-10, max_x1=10, min_x2=-10, max_x2=10):
    x1 = np.linspace(min_x1, max_x1, num=10*(max_x1-min_x1)).reshape(-1, 1)
    x2 = np.linspace(min_x2, max_x2, num=10*(max_x2-min_x2)).reshape(-1, 1)
    mesh = np.meshgrid(x1, x2, indexing='ij')
    p = np.zeros([len(x1), len(x2)])

    for i, u in enumerate(x1):
        for j, v in enumerate(x2):
            x = np.array([u, v]).reshape(-1, 1)
            dist_1 = sorted([np.linalg.norm(x-_.reshape(-1, 1)) for _ in samples_1])
            dist_2 = sorted([np.linalg.norm(x-_.reshape(-1, 1)) for _ in samples_2])
            n_1 = 0
            n_2 = 0
            while n_1+n_2 < k:
                if min(dist_1) < min(dist_2):
                    n_1 += 1
                    dist_1.pop(0)
                else:
                    n_2 += 1
                    dist_2.pop(0)
            p[i, j] = 0 if n_1 < n_2 else 1

    return mesh[0], mesh[1], p


def plot_3d(pdf_A, pdf_B, filename, scale=2e2):
    mlab.clf()
    mlab.surf(pdf_A, colormap='Reds', warp_scale=scale)
    mlab.surf(pdf_B, colormap='Blues', warp_scale=scale)
    mlab.view(distance=500, focalpoint=(3, 2, 0))
    # the figure is saved and displayed as an image
    # because the interactive version is not convertible to pdf
    mlab.savefig(filename=filename, size=(2000, 2000))


def plot_knn(mesh):
    pass


def plot_regions(pdf_A, pdf_B):
    diff_x1 = (np.diff(np.sign(pdf_A[2] - pdf_B[2]), axis=0, append=0) != 0)*1
    diff_x2 = (np.diff(np.sign(pdf_A[2] - pdf_B[2]), axis=1, append=0) != 0)*1
    change = (diff_x1 | diff_x2).T
    regions = (pdf_A[2] > pdf_B[2]).T*1

    fig, ax = plt.subplots(1, 2)
    ax[0].contour(*pdf_A, cmap=cm.Reds)
    ax[0].contour(*pdf_B, cmap=cm.Blues)
    ax[0].imshow(change, cmap=cm.binary, extent=[pdf_A[0][0][0], pdf_A[0]
                                                 [-1][-1], pdf_A[1][0][0], pdf_A[1][0][-1]], origin='lower')

    ax[1].contour(*pdf_A, cmap=cm.Reds)
    ax[1].contour(*pdf_B, cmap=cm.Blues)
    ax[1].imshow(regions, cmap=cm.seismic, extent=[pdf_A[0][0][0], pdf_A[0]
                                                   [-1][-1], pdf_A[1][0][0], pdf_A[1][0][-1]], origin='lower')


if __name__ == '__main__':
    samples_1, samples_2 = np.load('assignments/lab3/lab3.p', allow_pickle=True)
    knn_regions = knn(samples_1.T, samples_2.T, 3)
