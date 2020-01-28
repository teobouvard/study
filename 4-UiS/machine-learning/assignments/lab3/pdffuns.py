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
    pass

def knn(samples, k):
    pass

def plot_3d(pdf_A, pdf_B, filename):
    mlab.clf()
    mlab.surf(pdf_A, colormap='Reds', warp_scale=1e2)
    mlab.surf(pdf_B, colormap='Blues', warp_scale=1e2)
    mlab.view(distance=500, focalpoint=(3, 2, 0))
    # the figure is saved and displayed as an image
    # because the interactive version is not convertible to pdf
    mlab.savefig(filename=filename, size=(2000, 2000))

def plot_regions(pdf_A, pdf_B):
    change = ((np.diff(np.sign(pdf_A[2] - pdf_B[2])) != 0)*1).T
    regions = (pdf_A[2] > pdf_B[2]).T*1

    fig, ax = plt.subplots(1, 2)
    ax[0].contour(*pdf_A, cmap=cm.Reds)
    ax[0].contour(*pdf_B, cmap=cm.Blues)
    ax[0].imshow(change, cmap=cm.binary, extent=[pdf_A[0][0][0], pdf_A[0][-1][-1], pdf_A[1][0][0], pdf_A[1][0][-1]], origin='lower')

    ax[1].contour(*pdf_A, cmap=cm.Reds)
    ax[1].contour(*pdf_B, cmap=cm.Blues)
    ax[1].imshow(regions, cmap=cm.seismic, extent=[pdf_A[0][0][0], pdf_A[0][-1][-1], pdf_A[1][0][0], pdf_A[1][0][-1]], origin='lower')