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


def plot_node_age_by_frequency(list_of_ages, title, xlabel, ylabel, outfile, colorkey, fit):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    num_samples = np.max([x[2] for t in list_of_ages[0] for x in t.values()])

    labels = []
    colors = []
    for (lab, col) in colorkey:
        labels.append(lab)
        colors.append(col)

    for i, post in enumerate(list_of_ages):
        time_by_freq = np.zeros(num_samples + 1)
        logtime_by_freq = np.zeros(num_samples + 1)
        total_span = np.zeros(num_samples + 1)
        for post_by_tree in post:
            for (time, logtime, freq, span) in post_by_tree.values():
                time_by_freq[freq] += span * time
                logtime_by_freq[freq] += span * logtime
                total_span[freq] += span
        ax1.scatter(np.arange(num_samples + 1), time_by_freq / total_span, c=colors[i], label=labels[i], s=2)
        ax2.scatter(np.arange(num_samples + 1), logtime_by_freq / total_span, c=colors[i], label=labels[i], s=2)

    ax1.scatter(np.arange(num_samples + 1), fit[:, 0], s=3, c="black", label='theory')
    ax2.scatter(np.arange(num_samples + 1), fit[:, 1], s=3, c="black", label='theory')
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
blarg_blarg = pickle.load(open(f"{argv[1]}/dated/blarg_blarg.pickle", "rb"))

# --- propagated mutations, true topologies --- #

true_mcmc_prop = pickle.load(open(f"{argv[1]}/dated/true_chr1.mcmc.propagated.pickle", "rb"))
true_vari_prop = pickle.load(open(f"{argv[1]}/dated/true_chr1.dated.propagated.pickle", "rb"))
true_vari_local = pickle.load(open(f"{argv[1]}/dated/true_chr1.dated.local.pickle", "rb"))

# node age by frequency
plot_node_age_by_frequency(
    [simul_truth['nodes'], true_mcmc_prop['nodes'], true_vari_prop['nodes']], 
    'Average age by frequency\n(true topologies)', 
    ['Number of descendant tips', 'Number of descendant tips'], ['E[age]', 'E[log age]'],
    f"{argv[1]}/fig/node_age_by_freq.true_chr1.propagated.png", 
    colorkey=[("simulation", "blue"), ("mcmc-relate", "red"), ("ep-relate", "green")],
    fit=true_vari_local['prior'],
)

plot_node_age_by_frequency(
    [simul_truth['nodes'], true_mcmc_prop['nodes'], blah_blah['nodes']],
    'Average age by frequency\n(true topologies)', 
    ['Number of descendant tips', 'Number of descendant tips'], ['E[age]', 'E[log age]'],
    f"{argv[1]}/fig/node_age_by_freq.true_chr1.local.png", 
    colorkey=[("simulation", "blue"), ("mcmc-relate", "red"), ("ep-marginal", "green")],
    fit=true_vari_local['prior'],
)

plot_node_age_by_frequency(
    [simul_truth['nodes'], true_mcmc_prop['nodes'], blarg_blarg['nodes']],
    'Average age by frequency\n(true topologies)', 
    ['Number of descendant tips', 'Number of descendant tips'], ['E[age]', 'E[log age]'],
    f"{argv[1]}/fig/node_age_by_freq.true_chr1.tsdate.png", 
    colorkey=[("simulation", "blue"), ("mcmc-relate", "red"), ("ep-tsdate", "green")],
    fit=true_vari_local['prior'],
)

# single mutation ages, methods vs one another
plot_mutation_ages(
    simul_truth['mutations'], blarg_blarg['mutations'], "Variational vs true mutation ages\n(true topologies, tsdate)", 
    ["True mutation age", "True log mutation age"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.dated.tsdate.png",
)

plot_mutation_ages(
    simul_truth['mutations'], true_vari_prop['mutations'], "Variational vs true mutation ages\n(true topologies, propagated mutations)", 
    ["True mutation age", "True log mutation age"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.dated.propagated.png",
)

plot_mutation_ages(
    simul_truth['mutations'], true_mcmc_prop['mutations'], "MCMC vs true mutation ages\n(true topologies, 100 MCMC samples)", 
    ["True mutation age", "True log mutation age"], ["MCMC E[mutation age]", "MCMC E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.mcmc.propagated.png",
)

plot_mutation_ages(
    true_mcmc_prop['mutations'], true_vari_prop['mutations'], "Variational vs MCMC mutation ages\n(true topologies, propagated mutations v. 100 MCMC samples)", 
    ["MCMC E[mutation age]", "MCMC E[log mutation age]"], ["Variational E[mutation age]", "Variational E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.mcmc_dated.propagated.png",
)

plot_mutation_ages(
    blarg_blarg['mutations'], true_mcmc_prop['mutations'], "Variational vs MCMC mutation ages\n(true topologies, tsdate v. 100 MCMC samples)", 
    ["Variational E[mutation age]", "Variational E[log mutation age]"], ["MCMC E[mutation age]", "MCMC E[log mutation age]"], 
    f"{argv[1]}/fig/single_mutations.true_chr1.mcmc_dated.tsdate.png",
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

