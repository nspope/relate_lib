"""
Python mockup of variational dating for Relate trees

Instead of matching moments of sufficient statistics, use mean and variance
(less shit to reimplement, works nearly as well)
"""

import numpy
import tskit
import scipy.special
import argparse

# --- library --- #

def marginalize_over_ancestors(moments):
    """
    Recursively marginalize over extant ancestors, using results
    from Wuif and Donnelly 1998.
    """

def coalescent_prior(num_tips):
    """
    Calculate E[t | k], E[log t | k] where t is node age and k is number of
    descendant samples.
    """

    # get hypoexponential moments

    # tilt moments with population size history
    # TODO

    # marginalize over number of extant ancestors

    # convert to gamma


def node_moment_matching(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Gamma projection of expectation propagation surrogate:

        XXX
    """
    # calculate mean and variance
    f0, f1, f2 = ...
    f1 *= ... / f0
    f2 *= ... / f0
    t = ...
    tsq = ...

    # convert to gamma, natural parameterization
    rate = t / (tsq - t ** 2)
    shape = t * rate
    return np.array([shape - 1, -rate])

def date_relate(ts, mu):
    """
    Variational dating of a Relate tree sequence.
    """

    edge_meta = numpy.frombuffer(ts.tables.edges.metadata, "i4, i4, f4")
    edge_order = [t.edge(n) for t in ts.trees() for n in t.nodes(order='time_asc')]
    
    # conditional coalescent prior
    prior = coalescent_prior(ts.num_samples)
    posterior = np.zeros((ts.num_nodes, 2))
    for tree in ts.trees():
        for node in tree.nodes():
            posterior[node] = prior[tree.num_samples(node)]
    
    # incorporate leaves (assumes samples are all contemporary)
    for tree in ts.trees():
        for child in range(ts.num_samples):
            left, right, mutations = edge_meta[tree.edge(child)]
            posterior[parent] += [mutations, (left - right) * mu]
    
    # expectation propagation, up trees then down again
    leafward_message = np.zeros((ts.num_edges, 2))
    rootward_message = np.zeros((ts.num_edges, 2))
    for e in edge_order + edge_order[::-1]:
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
    posterior[:, 0] -= 1.0 # natural parameters to shape
    posterior[:, 1] *= -1.0 # natural parameters to rate

    # approximate mutation age distributions

    return posterior


# --- interface --- #

ts = tskit.load("deleteme.trees")


# compare against ...
# do shit
# yay





