"""
Python mockup of variational dating for Relate trees

Instead of matching moments of sufficient statistics, use mean and variance
(less to reimplement, works nearly as well)
"""

import tsdate
import numpy as np
import tskit
import scipy.special
from scipy.special import gammainc, gammaln, betaln, hyp2f1
import argparse

# --- lib --- #

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

    # project to gamma
    #shape = moments[:, 0] ** 2 / (moments[:, 1] - moments[:, 0] ** 2)
    #rate = moments[:, 0] / (moments[:, 1] - moments[:, 0] ** 2)
    shape = np.zeros(num_tips + 1)
    rate = np.zeros(num_tips + 1)
    for i in range(2, num_tips + 1):
        shape[i], rate[i] = tsdate.approx.approximate_gamma_kl(*sufficient_statistics[i])

    return np.column_stack([shape - 1, -rate]) # natural parameterization


def node_moment_matching(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    Gamma projection of expectation propagation surrogate:

        q\ij p_ij(t_i, t_j) = 
            Ga(t_i | a_i, b_i) Ga(t_j | a_j, b_j) \times
                Po(y_ij | mu_ij(t_i - t_j))

    Using the natural parameterization `a_i = shape_i - 1, b_i = -rate_i`
    """

    # KL minimization
    _, proj_i, proj_j = tsdate.approx.gamma_projection(a_i + 1, -b_i, a_j + 1, -b_j, int(y_ij), -mu_ij)
    proj_i[0] -= 1; proj_i[1] *= -1
    proj_j[0] -= 1; proj_j[1] *= -1

    return np.array(proj_i), np.array(proj_j)


def date_relate(ts, mu, epoch_start=None, epoch_size=None, num_itt=1):
    """
    Variational dating of a Relate tree sequence.
    """

    edge_meta = np.frombuffer(ts.tables.edges.metadata, "i4, i4, f4")
    edge_order = [t.edge(n) for t in ts.trees() for n in t.nodes(order='timeasc')]
    
    # conditional coalescent prior
    if epoch_start is None:
        epoch_start = np.array([0.])
        epoch_size = np.array([0.5 * ts.diversity() / mu])
    prior = coalescent_prior(ts.num_samples, epoch_start, epoch_size)
    posterior = np.zeros((ts.num_nodes, 2))
    for tree in ts.trees():
        for node in tree.nodes():
            posterior[node] = prior[tree.num_samples(node)]
    
    # incorporate leaves (assumes samples are all contemporary)
    for tree in ts.trees():
        for child in range(ts.num_samples):
            if tree.edge(child) != tskit.NULL:
                left, right, mutations = edge_meta[tree.edge(child)]
                posterior[tree.parent(child)] += [mutations, (left - right) * mu]
    
    # expectation propagation, up trees then down again
    leafward_message = np.zeros((ts.num_edges, 2))
    rootward_message = np.zeros((ts.num_edges, 2))
    for itt in range(num_itt):
        for e in edge_order + edge_order[::-1]:
            if e != tskit.NULL:
                parent, child = ts.edges_parent[e], ts.edges_child[e]
                left, right, mutations = edge_meta[e] 
                if child >= ts.num_samples:
                    edge_likelihood = [mutations, (left - right) * mu]
                    child_cavity = posterior[child] - leafward_message[e]
                    parent_cavity = posterior[parent] - rootward_message[e]
                    posterior[parent], posterior[child] = \
                        node_moment_matching(*parent_cavity, *child_cavity, *edge_likelihood)
                    leafward_message[e] = posterior[child] - child_cavity
                    rootward_message[e] = posterior[parent] - parent_cavity
    posterior[:, 0] += 1.0 # natural parameters to shape
    posterior[:, 1] *= -1.0 # natural parameters to rate

    # approximate mutation age distributions
    # TODO

    return posterior


# --- fit variational model --- #

from sys import argv

ts = tskit.load(f"{argv[1]}/relate_outputs/chr1.sample1.trees")
foo = date_relate(ts, 1.25e-8, np.array([0.]), np.array([20000.]), num_itt=1) #should converge in 1 iteration

# get relate MCMC samples into an array
sampled_times = np.zeros((ts.num_nodes, 100))
for i in range(1, 101):
    sampled_times[:,i-1] = tskit.load(f"{argv[1]}/relate_outputs/chr1.sample{i}.trees").nodes_time
mean_time = np.mean(sampled_times, axis=1)
mean_log_time = np.mean(np.log(sampled_times), axis=1)
std_time = np.std(sampled_times, axis=1)

# ------- figures

import matplotlib.pyplot as plt

#-------- per node variational vs mcmc
plt.hexbin(ts.tables.nodes.time[ts.num_samples:], foo[ts.num_samples:,0] / foo[ts.num_samples:,1], gridsize=50, mincnt=1, bins='log')
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.xlabel("MCMC age (realization)")
plt.ylabel("Variational E[age]")
plt.title("Single MCMC sample")
plt.savefig(f"{argv[1]}/dating_vs_realization.png")
plt.clf()

plt.hexbin(mean_time[ts.num_samples:], foo[ts.num_samples:,0] / foo[ts.num_samples:,1], gridsize=50, mincnt=1, bins='log')
plt.xlabel("MCMC age (mean)")
plt.ylabel("Variational E[age]")
plt.title("Mean of 100 MCMC samples")
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.savefig(f"{argv[1]}/dating_vs_mean.png")
plt.clf()

plt.hexbin(std_time[ts.num_samples:], np.sqrt(foo[ts.num_samples:,0] / foo[ts.num_samples:,1] ** 2), gridsize=50, mincnt=1, bins='log')
plt.xlabel("MCMC age (stddev)")
plt.ylabel("Variational sqrt(V[age])")
plt.title("StdDev of 100 MCMC samples")
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.savefig(f"{argv[1]}/dating_vs_stddev.png")
plt.clf()

plt.hexbin(np.log(ts.tables.nodes.time[ts.num_samples:]), scipy.special.digamma(foo[ts.num_samples:,0]) - np.log(foo[ts.num_samples:,1]), gridsize=50, mincnt=1, bins='log')
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.xlabel("MCMC log age (realization)")
plt.ylabel("Variational E[log age]")
plt.title("Single MCMC sample, log'd")
plt.savefig(f"{argv[1]}/dating_vs_realization_log.png")
plt.clf()

plt.hexbin(mean_log_time[ts.num_samples:], scipy.special.digamma(foo[ts.num_samples:,0]) - np.log(foo[ts.num_samples:,1]), gridsize=50, mincnt=1, bins='log')
plt.xlabel("MCMC log age (mean)")
plt.ylabel("Variational E[log age]")
plt.title("Mean log of 100 MCMC samples")
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.savefig(f"{argv[1]}/dating_vs_mean_log.png")
plt.clf()

# ------- compare global summary statistics

# expected sufficient statistics vs number of descendants
true_ts = tskit.load(f"{argv[1]}/chr1.trees")
true_age_by_numdesc = np.zeros(ts.num_samples + 1) #from true ts
true_log_age_by_numdesc = np.zeros(ts.num_samples + 1) #from true ts
true_span_by_numdesc = np.zeros(ts.num_samples + 1)
for tree in true_ts.trees():
    for node in tree.nodes():
        k = tree.num_samples(node)
        true_age_by_numdesc[k] += true_ts.nodes_time[node] * tree.span
        true_log_age_by_numdesc[k] += np.log(true_ts.nodes_time[node]) * tree.span
        true_span_by_numdesc[k] += tree.span
true_age_by_numdesc /= true_span_by_numdesc
true_log_age_by_numdesc /= true_span_by_numdesc

mcmc_age_by_numdesc = np.zeros(ts.num_samples + 1) #from mean of mcmc
vari_age_by_numdesc = np.zeros(ts.num_samples + 1) #from variational method
mcmc_log_age_by_numdesc = np.zeros(ts.num_samples + 1) #from mean of mcmc
vari_log_age_by_numdesc = np.zeros(ts.num_samples + 1) #from variational method
span_by_numdesc = np.zeros(ts.num_samples + 1)
for tree in ts.trees():
    for node in tree.nodes():
        k = tree.num_samples(node)
        mcmc_log_age_by_numdesc[k] += np.mean(np.log(sampled_times[node])) * tree.span
        vari_log_age_by_numdesc[k] += (scipy.special.digamma(foo[node, 0]) - np.log(foo[node, 1])) * tree.span
        mcmc_age_by_numdesc[k] += mean_time[node] * tree.span
        vari_age_by_numdesc[k] += foo[node, 0] / foo[node, 1] * tree.span
        span_by_numdesc[k] += tree.span
mcmc_age_by_numdesc /= span_by_numdesc
vari_age_by_numdesc /= span_by_numdesc
mcmc_log_age_by_numdesc /= span_by_numdesc
vari_log_age_by_numdesc /= span_by_numdesc

prior = coalescent_prior(true_ts.num_samples, np.array([0.]), np.array([20000.]))
plt.scatter(np.arange(0, ts.num_samples + 1), true_log_age_by_numdesc, s=1, c="black", label='sim')
plt.scatter(np.arange(0, ts.num_samples + 1), mcmc_log_age_by_numdesc, s=1, c="red", label='mcmc')
plt.scatter(np.arange(0, ts.num_samples + 1), vari_log_age_by_numdesc, s=1, c="blue", label='vari')
plt.scatter(np.arange(0, ts.num_samples + 1), scipy.special.digamma(prior[:,0] + 1) - np.log(-prior[:,1]), s=1, c="green", label='prior')
plt.xlabel("Number of descendants")
plt.ylabel("Expected log age")
plt.legend(loc='upper left')
plt.savefig(f"{argv[1]}/marginal_log_age.png")
plt.clf()

plt.scatter(np.arange(0, ts.num_samples + 1), true_age_by_numdesc, s=1, c="black", label='sim')
plt.scatter(np.arange(0, ts.num_samples + 1), mcmc_age_by_numdesc, s=1, c="red", label='mcmc')
plt.scatter(np.arange(0, ts.num_samples + 1), vari_age_by_numdesc, s=1, c="blue", label='vari')
plt.scatter(np.arange(0, ts.num_samples + 1), (prior[:,0] + 1)/(-prior[:,1]), s=1, c="green", label='prior')
plt.xlabel("Number of descendants")
plt.ylabel("Expected age")
plt.legend(loc='upper left')
plt.savefig(f"{argv[1]}/marginal_age.png")
plt.clf()

# coalescence rates
