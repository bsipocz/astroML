import numpy as np
import pymc3 as pm

# ----------------------------------------------------------------------
# Define basic linear model


def model(xi, theta, intercept):
    slope = np.tan(theta)
    return slope * xi + intercept


def simulation_logp(x):
    # log eq 110
    return np.log(np.exp(x) / (1+np.exp(2.75*x)))


def dist(x):
    # eq 110
    return np.exp(x) / (1+np.exp(2.75*x))


def simulation(size=50, low=-10, high=10, alpha=1, beta=0.5,
               epsilon=(0, 0.75)):
    eps = np.random.normal(epsilon[0], scale=epsilon[1], size=size)

    x = pm.Uniform('x', low, high)

    xi = dist(x)
    # eq 1
    eta = alpha + beta * xi + eps

    tau = np.std(xi)

    # measurement errors from scaled inverse chi2 with df=5
    sigma_x = 1 / np.random.chisquare(df=5, size=size)
    sigma_y = epsilon[1] / np.random.chisquare(df=5, size=size)

    return xi, eta


with pm.Model():
    # uniform prior on Pb, the fraction of bad points
    Pb = pm.Uniform('Pb', 0, 1.0, testval=0.1)

    # uniform prior on Yb, the centroid of the outlier distribution
    Yb = pm.Uniform('Yb', -10000, 10000, testval=0)

    # uniform prior on log(sigmab), the spread of the outlier distribution
    log_sigmab = pm.Uniform('log_sigmab', -10, 10, testval=5)

    inter = pm.Uniform('inter', -200, 400)
    theta = pm.Uniform('theta', -np.pi / 2, np.pi / 2, testval=np.pi / 4)

    y_mixture = pm.DensityDist('simul', logp=simulation_logp,
                               observed={'yi': yi, 'xi': xi})

    trace1 = pm.sample(draws=5000, tune=1000)


def maximum_likelihood_estimate():
    pass

# TODO: reproduce figure 3


# TODO: reproduce figure 4


# TODO: reproduce figure 5


# TODO: reproduce figure 6


# TODO: reproduce figure 7


# TODO: reproduce figure 8


# TODO: reproduce figure 9

# TODO: reproduce figure 10
