"""
Compare variational and MCMC posteriors against truth, by:

    1. Comparing point estimates for mutation ages
    2. Comparing genome-wide averages for mutation time, stratified by frequency
    3. Comparing coalescence rates

This is done for (A) inference with propagated mutations; (B) inference with only local mutations
"""

# --- plotting functions --- #

def plot_mutation_ages(truth, inferred, title, xlabel, ylabel, outfile, offset=0):
    # make lookup table: offset because relate mutations are shifted by 1bp from sim
    true_mut_ages = {pos-offset:vals for tree in truth for (pos, vals) in tree.items()}
    inferred_times = []
    true_times = []
    inferred_logtimes = []
    true_logtimes = []
    for infer_tree in inferred:
        for pos, (infer_time, infer_logtime, infr_desc) in infer_tree.items():
            true_time, true_logtime, true_desc = true_mut_ages[pos]
            inferred_times.append(infer_time)
            true_times.append(true_time)
            inferred_logtimes.append(infer_logtime)
            true_logtimes.append(true_logtime)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    ax1.hexbin(true_times, inferred_times, gridsize=50, mincnt=1, bins='log')
    ax1.axline((0,0), slope=1, c="black", linestyle="dashed")
    ax1.set_xlabel(f"{xlabel[0]}")
    ax1.set_ylabel(f"{ylabel[0]}")
    ax2.hexbin(true_logtimes, inferred_logtimes, gridsize=50, mincnt=1, bins='log')
    ax2.axline((0,0), slope=1, c="black", linestyle="dashed")
    ax2.set_xlabel(f"{xlabel[1]}")
    ax2.set_ylabel(f"{ylabel[1]}")
    fig.savefig(outfile)
    fig.clf()


def plot_node_age_by_frequency(list_of_ages, title, xlabel, ylabel, outfile, fit):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    num_samples = np.max([x[2] for t in list_of_ages[0] for x in t.values()])

    labels = ['sim', 'mcmc', 'vari'] #TODO make argument (dict)
    cols = ['black', 'red', 'blue', 'green'] #TODO make argument (dict)

    for i, post in enumerate(list_of_ages):
        time_by_freq = np.zeros(num_samples + 1)
        logtime_by_freq = np.zeros(num_samples + 1)
        total_span = np.zeros(num_samples + 1)
        for post_by_tree in post:
            for (time, logtime, freq, span) in post_by_tree.values():
                time_by_freq[freq] += span * time
                logtime_by_freq[freq] += span * logtime
                total_span[freq] += span
        ax1.scatter(np.arange(num_samples + 1), time_by_freq / total_span, c=cols[i], label=labels[i], s=2)
        ax2.scatter(np.arange(num_samples + 1), logtime_by_freq / total_span, c=cols[i], label=labels[i], s=2)

    ax1.scatter(np.arange(num_samples + 1), fit[:, 0], s=2, c="purple", label='prior')
    ax2.scatter(np.arange(num_samples + 1), fit[:, 1], s=2, c="purple", label='prior')
    ax1.legend(loc='upper left')
    ax1.set_xlabel(f"{xlabel[0]}")
    ax1.set_ylabel(f"{ylabel[0]}")
    ax2.set_xlabel(f"{xlabel[1]}")
    ax2.set_ylabel(f"{ylabel[1]}")
    fig.savefig(outfile)
    fig.clf()
    

# --- make figures --- #

from sys import argv
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

if not os.path.exists(f"{argv[1]}/fig"):
    os.makedirs(f"{argv[1]}/fig")

simul_truth = pickle.load(open(f"{argv[1]}/dated/chr1.simul.pickle", "rb"))
blah_blah = pickle.load(open(f"{argv[1]}/dated/blah_blah.pickle", "rb"))

# --- propagated mutations, true topologies --- #

true_mcmc_prop = pickle.load(open(f"{argv[1]}/dated/true_chr1.mcmc.propagated.pickle", "rb"))
true_vari_prop = pickle.load(open(f"{argv[1]}/dated/true_chr1.dated.propagated.pickle", "rb"))
true_vari_local = pickle.load(open(f"{argv[1]}/dated/true_chr1.dated.local.pickle", "rb"))

plot_node_age_by_frequency(
    [simul_truth['nodes'], true_mcmc_prop['nodes'], true_vari_prop['nodes']], 
    'foobar', ['foo', 'bar'], ['foo', 'bar'],
    f"{argv[1]}/fig/node_age_by_freq.true_chr1.propagated.png", 
    fit=true_vari_local['prior'],
)
plot_node_age_by_frequency(
    [simul_truth['nodes'], true_mcmc_prop['nodes'], blah_blah['nodes']],
    'foobar', ['foo', 'bar'], ['foo', 'bar'],
    f"{argv[1]}/fig/blah_blah.png", 
    fit=true_vari_local['prior'],
)

plot_mutation_ages(
    simul_truth['mutations'], true_vari_prop['mutations'], "Variational vs true mutation ages\n(true topologies, propagated mutations)", 
    ["True mutation age", "True log mutation age"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.dated.propagated.png",
)

plot_mutation_ages(
    simul_truth['mutations'], true_mcmc_prop['mutations'], "MCMC vs true mutation ages\n(true topologies, propagated mutations, 100 MCMC samples)", 
    ["True mutation age", "True log mutation age"], ["MCMC E[mutation age]", "MCMC E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.mcmc.propagated.png",
)

plot_mutation_ages(
    true_mcmc_prop['mutations'], true_vari_prop['mutations'], "Variational vs MCMC mutation ages\n(true topologies, propagated mutations, 100 MCMC samples)", 
    ["MCMC E[mutation age]", "MCMC E[log mutation age]"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.mcmc_dated.propagated.png",
)

# --- propagated mutations, inferred topologies --- #

rela_mcmc_prop = pickle.load(open(f"{argv[1]}/dated/chr1.mcmc.propagated.pickle", "rb"))
rela_vari_prop = pickle.load(open(f"{argv[1]}/dated/chr1.dated.propagated.pickle", "rb"))

plot_mutation_ages(
    simul_truth['mutations'], rela_vari_prop['mutations'], "Variational vs true mutation ages\n(inferred topologies, propagated mutations)", 
    ["True mutation age", "True log mutation age"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.chr1.dated.propagated.png", offset=1 #relate mutation positions off by 1
)

plot_mutation_ages(
    simul_truth['mutations'], rela_mcmc_prop['mutations'], "MCMC vs true mutation ages\n(inferred topologies, propagated mutations, 100 MCMC samples)", 
    ["True mutation age", "True log mutation age"], ["MCMC E[mutation age]", "MCMC E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.chr1.mcmc.propagated.png", offset=1 #relate mutation positions off by 1
)

plot_mutation_ages(
    rela_mcmc_prop['mutations'], rela_vari_prop['mutations'], "Variational vs MCMC mutation ages\n(inferred topologies, propagated mutations, 100 MCMC samples)", 
    ["MCMC E[mutation age]", "MCMC E[log mutation age]"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.chr1.mcmc_dated.propagated.png",
)

