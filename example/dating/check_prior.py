"""
Check marginal coalescent prior against simulation
"""

# --- lib --- #

import tsdate
import numpy as np
import tskit
import scipy.special
from scipy.special import gammainc, gammaln, betaln, hyp2f1, digamma

def tilt_moments_by_coalrate(mean, variance, epoch_start, epoch_size):
    """
    Given mean/variance for the distribution of age of a node: project to gamma
    distribution, tilt by haploid population size (e.g. rescale time by
    coalescent rate), and recalculate mean and variance under the tilted
    distribution.
    """

    # gamma projection
    shape = mean ** 2 / variance
    rate = mean / variance

    # convert epoch breakpoints to coalescent units
    step = np.cumsum(epoch_start[1:] * (1.0 / epoch_size[:-1] - 1.0 / epoch_size[1:]))
    coal_start = epoch_start / epoch_size + np.pad(step, (1, 0))
    cdf_breaks = np.append(coal_start, [np.inf])

    # piecewise integration along population size trajectory
    cdf_0 = np.diff(gammainc(shape, rate * cdf_breaks))
    mn_coef_0 = epoch_start - epoch_size * coal_start
    va_coef_0 = mn_coef_0 ** 2
    cdf_1 = mean * np.diff(gammainc(shape + 1, rate * cdf_breaks))
    mn_coef_1 = epoch_size
    va_coef_1 = mn_coef_0 * mn_coef_1 * 2
    cdf_2 = (shape + 1) * variance * np.diff(gammainc(shape + 2, rate * cdf_breaks))
    mn_coef_2 = 0.0
    va_coef_2 = mn_coef_1 ** 2

    # recalculate mean and variance
    tilted_mean = np.sum(mn_coef_2 * cdf_2 + mn_coef_1 * cdf_1 + mn_coef_0 * cdf_0)
    tilted_variance = np.sum(va_coef_2 * cdf_2 + va_coef_1 * cdf_1 + va_coef_0 * cdf_0)
    tilted_variance -= tilted_mean**2

    return tilted_mean, tilted_variance


def marginalize_over_ancestors(val):
    """
    Integrate an expectation over counts of extant ancestors. In a tree with
    "n" tips, the probability that there are "a" extent ancestors when a
    subtree of size "k" coalesces is hypergeometric-ish (Wuif & Donnelly 1998),
    and may be calculated recursively over increasing "a" and decreasing "k"
    (e.g. using recursive relationships for binomial coefficients).
    """
    n, N = val.shape  # number of tips, number of moments
    pr_a_ln = [np.nan, np.nan, 0.0]  # log Pr(a | k, n)
    out = np.zeros((n + 1, N))
    for k in range(n - 1, 1, -1):
        const = np.log(n - k) + np.log(k - 2) - np.log(k + 1) #TODO warning when k==1
        for a in range(2, n - k + 2):
            out[k] += np.exp(pr_a_ln[a]) * val[a]
            if k > 2:  # Pr(a | k, n) to Pr(a | k - 1, n)
                pr_a_ln[a] += const - np.log(n - a - k + 2)
        if k > 2:  # Pr(n - k + 1 | k - 1, n) to Pr(n - k + 2 | k - 1, n)
            pr_a_ln.append(pr_a_ln[-1] + np.log(n - k + 2) - np.log(k + 1) - const)
    out[n] = val[1]
    return out


def coalescent_prior(num_tips, epoch_start, epoch_size):
    """
    Calculate E[t | k], V[t | k] where t is node age and k is number of
    descendant samples.
    """

    coal_rates = np.array(
        [2 / (i * (i - 1)) if i > 1 else 0.0 for i in range(1, num_tips + 1)]
    )

    # hypoexponential mean and variance; e.g. conditional on the number of
    # extant ancestors when the node coalesces, the expected time of
    # coalescence is the sum of exponential RVs (Wuif and Donnelly 1998)
    mean = coal_rates.copy()
    variance = coal_rates.copy() ** 2
    for i in range(coal_rates.size - 2, 0, -1):
        mean[i] += mean[i + 1]
        variance[i] += variance[i + 1]

    # tilt moments with population size history
    for i in range(1, mean.size):
        mean[i], variance[i] = tilt_moments_by_coalrate(
            mean[i], variance[i], epoch_start, epoch_size
        )

    # Taylor approximate log expectation
    sufficient_statistics = np.column_stack([
        mean, np.log(mean) - 1/2 * variance / mean**2,
    ])

    # marginalize over number of extant ancestors
    #moments = marginalize_over_ancestors(np.stack((mean, variance + mean**2), 1))
    sufficient_statistics = marginalize_over_ancestors(sufficient_statistics)
    return sufficient_statistics

    # project to gamma
    #shape = moments[:, 0] ** 2 / (moments[:, 1] - moments[:, 0] ** 2)
    #rate = moments[:, 0] / (moments[:, 1] - moments[:, 0] ** 2)
    shape = np.zeros(num_tips + 1)
    rate = np.zeros(num_tips + 1)
    for i in range(2, num_tips + 1):
        shape[i], rate[i] = tsdate.approx.approximate_gamma_kl(*sufficient_statistics[i])

    return np.column_stack([shape - 1, -rate]) # natural parameterization

# --- check against simulation --- #

import msprime
import matplotlib.pyplot as plt

ts = msprime.sim_ancestry(
  samples=100,
  ploidy=1,
  recombination_rate=1e-8,
  population_size=20000,
  sequence_length=10e6,
  random_seed=1088,
)
ts = msprime.sim_mutations(
  ts, rate=1.25e-8, random_seed=1024
)

time = np.zeros(ts.num_samples + 1)
logtime = np.zeros(ts.num_samples + 1)
span = np.zeros(ts.num_samples + 1)
for tree in ts.trees():
    for node in tree.nodes():
        k = tree.num_samples(node)
        time[k] += ts.nodes_time[node] * tree.span
        logtime[k] += np.log(ts.nodes_time[node]) * tree.span
        span[k] += tree.span
time /= span
logtime /= span

prior = coalescent_prior(ts.num_samples, np.array([0.]), np.array([20000.]))

plt.scatter(time[2:], prior[2:,0], s=2)
plt.axline((0,0), slope=1, c="black")
plt.savefig("prior_check_1.png")
plt.clf()

plt.scatter(np.arange(2, ts.num_samples+1), prior[2:,0], s=2, c="blue")
plt.scatter(np.arange(2, ts.num_samples+1), time[2:], s=2, c="red")
plt.savefig("prior_check_2.png")
plt.clf()

plt.scatter(logtime[2:], prior[2:,1], s=2)
plt.axline((0,0), slope=1, c="black")
plt.savefig("prior_check_3.png")
plt.clf()

plt.scatter(np.arange(2, ts.num_samples+1), prior[2:,1], s=2, c="blue")
plt.scatter(np.arange(2, ts.num_samples+1), logtime[2:], s=2, c="red")
plt.savefig("prior_check_4.png")
plt.clf()


