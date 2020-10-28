# Based on figure 6.11, extreme Deconvolution example

import os


def compute_XD_results(n_points=2000, n_components=10, max_iter=500, threading=False):

    if not threading:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    # moving imports here, so threading can be adjusted from the function
    import dask.array as da
    import numpy as np
    from astroML.density_estimation import XDGMM

    # ------------------------------------------------------------
    # Sample the dataset
    da.random.seed(0)

    # generate the true data
    x_true = (1.4 + 2 * da.random.random(n_points)) ** 2
    y_true = 0.1 * x_true ** 2

    # add scatter to "true" distribution
    dx = 0.1 + 4. / x_true ** 2
    dy = 0.1 + 10. / x_true ** 2

    x_true += da.random.normal(0, dx, n_points)
    y_true += da.random.normal(0, dy, n_points)

    # add noise to get the "observed" distribution
    dx = 0.2 + 0.5 * da.random.random(n_points)
    dy = 0.2 + 0.5 * da.random.random(n_points)

    x = x_true + da.random.normal(0, dx)
    y = y_true + da.random.normal(0, dy)

    # stack the results for computation
    X = da.vstack([x, y]).T

    # dask.array doesn't yet support item asignment, thus now falling back on numpy
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([dx ** 2, dy ** 2]).T

    Xerr = da.from_array(Xerr)

    clf = XDGMM(n_components, max_iter=max_iter)
    clf.fit(X, Xerr)
    sample = clf.sample(n_points)



