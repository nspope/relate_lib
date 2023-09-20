"""
Python mockup of variational dating for Relate trees

Dumps lists containing a dict for each marginal tree, with posterior summaryu
for 
"""

import tsdate
import numpy as np
import tskit
import scipy.special
from scipy.special import gammainc, gammaln, betaln, hyp2f1, digamma

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
    _, proj_i, proj_j = tsdate.approx.gamma_projection(
        a_i + 1, -b_i, a_j + 1, -b_j, int(y_ij), -mu_ij
    )
    proj_i[0] -= 1; proj_i[1] *= -1
    proj_j[0] -= 1; proj_j[1] *= -1

    return np.array(proj_i), np.array(proj_j)


def mutation_moment_matching(a_i, b_i, a_j, b_j, y_ij, mu_ij, leaf):
    r"""
    Gamma projection of expectation propagation surrogate:

        q\ij p_ij(t_ij, t_i, t_j) = 
            Ga(t_i | a_i, b_i) Ga(t_j | a_j, b_j) \times
                Po(y_ij | mu_ij(t_i - t_j)) \times
                    Uniform(t_ij | t_j, t_i)

    Using the natural parameterization `a_i = shape_i - 1, b_i = -rate_i`
    """

    # KL minimization
    proj_ij = tsdate.approx.mutation_gamma_projection(
        a_i + 1, -b_i, a_j + 1, -b_j, int(y_ij), -mu_ij, leaf=leaf
    )
    proj_ij[0] -= 1; proj_ij[1] *= -1

    return np.array(proj_ij)


def date_relate(ts, mu, epoch_start, epoch_size, propagate_mutations=True, num_itt=1):
    """
    Variational dating of a Relate tree sequence.
    """

    if propagate_mutations: 
        # placed into metadata by relate_lib::Convert
        metadata = np.frombuffer(ts.tables.edges.metadata, "i4, i4, f4")
        edge_meta = [
            (metadata[e][0], metadata[e][1], int(metadata[e][2])) for e in range(ts.num_edges)
        ]
    else:
        # mutations occuring within tree span
        edge_mutations = np.zeros(ts.num_edges)
        for m in ts.mutations():
            if m.edge != tskit.NULL:
                edge_mutations[m.edge] += 1.0
        edge_meta = [
            (ts.edges_left[e], ts.edges_right[e], edge_mutations[e]) for e in range(ts.num_edges)
        ]

    node_prior = coalescent_prior(ts.num_samples, epoch_start, epoch_size)
    node_posterior = []
    muts_posterior = []

    for tree in ts.trees():
        if tree.num_edges:
            assert tree.num_edges == 2 * tree.num_samples() - 2 # assume binary

        edge_order = [
            tree.edge(n) for n in tree.nodes(order='postorder') 
            if tree.edge(n) != tskit.NULL
        ]
        leafward_message = {e:[0, 0] for e in edge_order}
        rootward_message = {e:[0, 0] for e in edge_order}

        muts = {}
        edges = {}
        nodes = {}
    
        # conditional coalescent prior
        for node in tree.nodes():
            nodes[node] = node_prior[tree.num_samples(node)].copy()

        # incorporate leaves (assumes samples are all contemporary)
        for child in range(ts.num_samples):
            parent, edge = tree.parent(child), tree.edge(child)
            if edge != tskit.NULL:
                left, right, mutations = edge_meta[edge]
                rootward_message[edge] = [mutations, (left - right) * mu]
                nodes[parent] += rootward_message[edge]
    
        # expectation propagation, up trees then down again
        for itt in range(num_itt):
            for e in edge_order + edge_order[::-1]:
                left, right, mutations = edge_meta[e] 
                parent, child = ts.edges_parent[e], ts.edges_child[e]
                if child >= ts.num_samples:
                    edge_likelihood = [mutations, (left - right) * mu]
                    parent_cavity = nodes[parent] - rootward_message[e]
                    child_cavity = nodes[child] - leafward_message[e]
                    nodes[parent], nodes[child] = \
                        node_moment_matching(*parent_cavity, *child_cavity, *edge_likelihood)
                    rootward_message[e] = nodes[parent] - parent_cavity
                    leafward_message[e] = nodes[child] - child_cavity

        # mutation age distributions
        for e in edge_order:
            left, right, mutations = edge_meta[e] 
            parent, child = ts.edges_parent[e], ts.edges_child[e]
            if mutations > 0:
                is_a_leaf = child < ts.num_samples
                edge_likelihood = [mutations, (left - right) * mu]
                edges[e] = mutation_moment_matching(
                    *(nodes[parent] - rootward_message[e]),
                    *(nodes[child] - leafward_message[e]),
                    *edge_likelihood, is_a_leaf
                )

        # convert from natural parameterization to sufficient statistics
        for node in tree.nodes():
            alpha, beta = nodes.pop(node)
            num_desc = tree.num_samples(node)
            if node >= ts.num_samples:
                nodes[node] = ((alpha + 1)/-beta, digamma(alpha + 1) - np.log(-beta), num_desc, tree.span)
        node_posterior.append(nodes)

        for mut in tree.mutations():
            alpha, beta = edges[mut.edge]
            position = ts.sites_position[mut.site]
            num_desc = tree.num_samples(mut.node)
            muts[position] = ((alpha + 1)/-beta, digamma(alpha + 1) - np.log(-beta), num_desc)
        muts_posterior.append(muts)

    node_prior[:2] = [0.0, -np.inf]
    for num_desc in range(2, ts.num_samples + 1):
        alpha, beta = node_prior[num_desc]
        node_prior[num_desc] = [(alpha + 1)/-beta, digamma(alpha + 1) - np.log(-beta)]

    # format per node/mutation: (expected_time, expected_logtime, number_of_descendant_samples)
    return {"nodes":node_posterior, "mutations":muts_posterior, "prior":node_prior}


# --- fit variational model, make list of dicts with nodes/mutations per marginal tree --- #

import os
from sys import argv
import pickle
import glob

if not os.path.exists(f"{argv[1]}/dated"):
    os.makedirs(f"{argv[1]}/dated")

Ne = 20000.

# relate-inferred tree sequence
rela_ts = tskit.load(f"{argv[1]}/relate_outputs/chr1.sample1.trees")
#rela_local = date_relate(
#    rela_ts, 1.25e-8, np.array([0.]), np.array([Ne]), propagate_mutations=False, num_itt=1
#)
#rela_prop = date_relate(
#    rela_ts, 1.25e-8, np.array([0.]), np.array([Ne]), propagate_mutations=True, num_itt=1
#)
#pickle.dump(rela_local, open(f"{argv[1]}/dated/chr1.dated.local.pickle", "wb"))
#pickle.dump(rela_prop, open(f"{argv[1]}/dated/chr1.dated.propagated.pickle", "wb"))

# relatified simulation (propagate mutations according to true edges)
true_ts = tskit.load(f"{argv[1]}/relate_outputs/true_chr1.sample1.trees")
#true_local = date_relate(
#    true_ts, 1.25e-8, np.array([0.]), np.array([Ne]), propagate_mutations=False, num_itt=1
#)
#true_prop = date_relate(
#    true_ts, 1.25e-8, np.array([0.]), np.array([Ne]), propagate_mutations=True, num_itt=1
#)
#pickle.dump(true_local, open(f"{argv[1]}/dated/true_chr1.dated.local.pickle", "wb"))
#pickle.dump(true_prop, open(f"{argv[1]}/dated/true_chr1.dated.propagated.pickle", "wb"))

# compare against tsdate implementation of algo
true_ts_nometa = true_ts.dump_tables()
true_ts_nometa.edges.packset_metadata([b''] * true_ts_nometa.edges.num_rows)
#true_ts_nometa.delete_intervals([[[x for x in true_ts.breakpoints()][100], true_ts.sequence_length]])
#true_ts_nometa.trim()
true_ts_nometa = true_ts_nometa.tree_sequence()
prior = coalescent_prior(true_ts_nometa.num_samples, np.array([0.]), np.array([20000.]))
grid = tsdate.prior.MixturePrior(true_ts_nometa, prior_distribution="gamma").make_parameter_grid(population_size=10000.)
for tree in true_ts_nometa.trees():
    for node in tree.nodes():
        if node >= true_ts_nometa.num_samples:
            alpha, beta = prior[tree.num_samples(node)]
            grid[node] = [alpha + 1, -beta]
blah, baz = tsdate.date(
    true_ts_nometa, mutation_rate=1.25e-8, 
    #population_size=1e4, global_prior=True,
    priors=grid, global_prior=False, 
    method="variational_gamma", max_iterations=1, return_posteriors=True, progress=True
)

# compare against FULL tsdate implementation
simul_ts = tskit.load(f"{argv[1]}/chr1.trees")
blarg = tsdate.date(simul_ts, population_size=10000., mutation_rate=1.25e-8, method="variational_gamma", max_iterations=10, progress=True)

# --- aggregate relate MCMC output into the same per-tree format --- #

def aggregate_mcmc(ts, samples):
    node_times = np.full((ts.num_nodes, len(samples)), np.nan)
    mutation_times = np.full((ts.num_mutations, len(samples)), np.nan)

    above_mut = np.array([ts.edges_parent[mut.edge] for mut in ts.mutations()])
    below_mut = np.array([ts.edges_child[mut.edge] for mut in ts.mutations()])
    unmapped = np.array([mut.edge == tskit.NULL for mut in ts.mutations()])

    for i in range(1, len(samples) + 1):
        node_times[:, i-1] = tskit.load(f"{samples[i-1]}").nodes_time
        mutation_times[:, i-1] = node_times[above_mut, i-1] / 2 + node_times[below_mut, i-1] / 2
    mutation_times[unmapped, :] = np.nan

    node_logtimes = np.mean(np.log(node_times), axis=1)
    mutation_logtimes = np.mean(np.log(mutation_times), axis=1)
    node_times = np.mean(node_times, axis=1)
    mutation_times = np.mean(mutation_times, axis=1)

    node_posterior = []
    muts_posterior = []
    for tree in ts.trees():
        nodes = {}
        muts = {}
        for node in tree.nodes():
            num_desc = tree.num_samples(node)
            nodes[node] = (node_times[node], node_logtimes[node], num_desc, tree.span)
        node_posterior.append(nodes)
        for mut in tree.mutations():
            pos = ts.sites_position[mut.site]
            num_desc = tree.num_samples(mut.node)
            muts[pos] = (mutation_times[mut.id], mutation_logtimes[mut.id], num_desc)
        muts_posterior.append(muts)

    # format per node/mutation: (expected_time, expected_logtime, number_of_descendant_samples)
    return {"nodes":node_posterior, "mutations":muts_posterior}


rela_mcmc = aggregate_mcmc(rela_ts, glob.glob(f"{argv[1]}/relate_outputs/chr1.sample*trees"))
true_mcmc = aggregate_mcmc(true_ts, glob.glob(f"{argv[1]}/relate_outputs/true_chr1.sample*trees"))
pickle.dump(rela_mcmc, open(f"{argv[1]}/dated/chr1.mcmc.propagated.pickle", "wb"))
pickle.dump(true_mcmc, open(f"{argv[1]}/dated/true_chr1.mcmc.propagated.pickle", "wb"))


# --- aggregate true ages (e.g. from simulation) into same format --- #

def aggregate_truth(ts):
    node_times = np.full(ts.num_nodes, np.nan)
    mutation_times = np.full(ts.num_mutations, np.nan)

    above_mut = np.array([ts.edges_parent[mut.edge] for mut in ts.mutations()])
    below_mut = np.array([ts.edges_child[mut.edge] for mut in ts.mutations()])
    unmapped = np.array([mut.edge == tskit.NULL for mut in ts.mutations()])

    node_truth = []
    muts_truth = []
    for tree in ts.trees():
        nodes = {}
        muts = {}
        for node in tree.nodes():
            num_desc = tree.num_samples(node)
            nodes[node] = (ts.nodes_time[node], np.log(ts.nodes_time[node]), num_desc, tree.span)
        for mut in tree.mutations():
            pos = ts.sites_position[mut.site]
            num_desc = tree.num_samples(mut.node)
            muts[pos] = (mut.time, np.log(mut.time), num_desc)
        node_truth.append(nodes)
        muts_truth.append(muts)

    # format per node/mutation: (true_time, true_logtime, number_of_descendant_samples)
    return {"nodes":node_truth, "mutations":muts_truth}


simul_ts = tskit.load(f"{argv[1]}/chr1.trees")
simul_truth = aggregate_truth(simul_ts)
pickle.dump(simul_truth, open(f"{argv[1]}/dated/chr1.simul.pickle", "wb"))

blah_blah = aggregate_truth(blah)
pickle.dump(blah_blah, open(f"{argv[1]}/dated/blah_blah.pickle", "wb"))

blarg_blarg = aggregate_truth(blarg)
pickle.dump(blarg_blarg, open(f"{argv[1]}/dated/blarg_blarg.pickle", "wb"))

assert False


# ------- figures

import matplotlib.pyplot as plt

#-------- per node variational vs mcmc
plt.hexbin(ts.tables.nodes.time[ts.num_samples:], foo[ts.num_samples:,0] / foo[ts.num_samples:,1], gridsize=50, mincnt=1, bins='log')
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.xlabel("MCMC age (realization)")
plt.ylabel("Variational E[age]")
plt.title("Single MCMC sample")
plt.savefig(f"{argv[1]}/fig/dating_vs_realization.png")
plt.clf()

plt.hexbin(mean_time[ts.num_samples:], foo[ts.num_samples:,0] / foo[ts.num_samples:,1], gridsize=50, mincnt=1, bins='log')
plt.xlabel("MCMC age (mean)")
plt.ylabel("Variational E[age]")
plt.title("Mean of 100 MCMC samples")
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.savefig(f"{argv[1]}/fig/dating_vs_mean.png")
plt.clf()

plt.hexbin(std_time[ts.num_samples:], np.sqrt(foo[ts.num_samples:,0] / foo[ts.num_samples:,1] ** 2), gridsize=50, mincnt=1, bins='log')
plt.xlabel("MCMC age (stddev)")
plt.ylabel("Variational sqrt(V[age])")
plt.title("StdDev of 100 MCMC samples")
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.savefig(f"{argv[1]}/fig/dating_vs_stddev.png")
plt.clf()

plt.hexbin(np.log(ts.tables.nodes.time[ts.num_samples:]), scipy.special.digamma(foo[ts.num_samples:,0]) - np.log(foo[ts.num_samples:,1]), gridsize=50, mincnt=1, bins='log')
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.xlabel("MCMC log age (realization)")
plt.ylabel("Variational E[log age]")
plt.title("Single MCMC sample, log'd")
plt.savefig(f"{argv[1]}/fig/dating_vs_realization_log.png")
plt.clf()

plt.hexbin(mean_log_time[ts.num_samples:], scipy.special.digamma(foo[ts.num_samples:,0]) - np.log(foo[ts.num_samples:,1]), gridsize=50, mincnt=1, bins='log')
plt.xlabel("MCMC log age (mean)")
plt.ylabel("Variational E[log age]")
plt.title("Mean log of 100 MCMC samples")
plt.axline((0,0), slope=1, c="black", linestyle="dashed")
plt.savefig(f"{argv[1]}/fig/dating_vs_mean_log.png")
plt.clf()

# ------- compare global summary statistics

# expected sufficient statistics vs number of descendants
#sim_ts = true_ts

#for simulation
sim_age_by_numdesc = np.zeros(ts.num_samples + 1) #from sim ts
true_age_by_numdesc = np.zeros(ts.num_samples + 1) #from sim ts
sim_log_age_by_numdesc = np.zeros(ts.num_samples + 1) #from sim ts
true_log_age_by_numdesc = np.zeros(ts.num_samples + 1) #from sim ts
true_span_by_numdesc = np.zeros(ts.num_samples + 1)
for tree in sim_ts.trees():
    for node in tree.nodes():
        k = tree.num_samples(node)
        sim_age_by_numdesc[k] += sim_ts.nodes_time[node] * tree.span
        sim_log_age_by_numdesc[k] += np.log(sim_ts.nodes_time[node]) * tree.span
        true_log_age_by_numdesc[k] += (scipy.special.digamma(bar[node, 0]) - np.log(bar[node, 1])) * tree.span
        true_age_by_numdesc[k] += bar[node, 0] / bar[node, 1] * tree.span
        true_span_by_numdesc[k] += tree.span
sim_age_by_numdesc /= true_span_by_numdesc
sim_log_age_by_numdesc /= true_span_by_numdesc
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

prior = coalescent_prior(sim_ts.num_samples, np.array([0.]), np.array([20000.]))
plt.scatter(np.arange(0, ts.num_samples + 1), sim_log_age_by_numdesc, s=1, c="black", label='sim')
plt.scatter(np.arange(0, ts.num_samples + 1), mcmc_log_age_by_numdesc, s=1, c="red", label='mcmc')
plt.scatter(np.arange(0, ts.num_samples + 1), vari_log_age_by_numdesc, s=1, c="blue", label='vari')
plt.scatter(np.arange(0, ts.num_samples + 1), scipy.special.digamma(prior[:,0] + 1) - np.log(-prior[:,1]), s=1, c="green", label='prior')
plt.xlabel("Number of descendants")
plt.ylabel("Expected log age")
plt.legend(loc='upper left')
plt.savefig(f"{argv[1]}/fig/marginal_log_age.png")
plt.clf()

plt.scatter(np.arange(0, ts.num_samples + 1), sim_age_by_numdesc, s=1, c="black", label='sim')
plt.scatter(np.arange(0, ts.num_samples + 1), mcmc_age_by_numdesc, s=1, c="red", label='mcmc')
plt.scatter(np.arange(0, ts.num_samples + 1), vari_age_by_numdesc, s=1, c="blue", label='vari')
plt.scatter(np.arange(0, ts.num_samples + 1), (prior[:,0] + 1)/(-prior[:,1]), s=1, c="green", label='prior')
plt.xlabel("Number of descendants")
plt.ylabel("Expected age")
plt.legend(loc='upper left')
plt.savefig(f"{argv[1]}/fig/marginal_age.png")
plt.clf()

# w/ true topologies
plt.scatter(np.arange(0, ts.num_samples + 1), sim_age_by_numdesc, s=1, c="black", label='sim')
plt.scatter(np.arange(0, ts.num_samples + 1), true_age_by_numdesc, s=1, c="red", label='vari-true')
plt.scatter(np.arange(0, ts.num_samples + 1), vari_age_by_numdesc, s=1, c="blue", label='vari-infr')
plt.scatter(np.arange(0, ts.num_samples + 1), (prior[:,0] + 1)/(-prior[:,1]), s=1, c="green", label='prior')
plt.xlabel("Number of descendants")
plt.ylabel("Expected age")
plt.legend(loc='upper left')
plt.savefig(f"{argv[1]}/marginal_age_true_topology.png")
plt.clf()

# coalescence rates
