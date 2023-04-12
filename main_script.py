# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:52:55 2022

This is the main script of this repository. It can be used to reproduce the
results presented in "Assortative mixing in micro-architecturally annotated
brain connectomes".

The computation of some of the results presented in the paper takes a
long time. Pre-computed results are therefore saved in the `results/`
directory of this repository.

The script is split into different cells (separated by `#%%`). Each cell
represent lines of codes used to load the data, compute a specific result, or
plot the elements of a specific figure:

DATA: Load data used in the experiments

RESULT 3: Standardized assortativity of micro-architectural annotations
RESULT 4: Assortative mixing of long-range connections
RESULTS 5: Heterophilic mixing
RESULTS 6: Homophilic ratios
RESULTS S1: Assortativity relative to spatially-naive nulls
RESULTS S2: Sensitivity and replication (homophilic mixing) | SC
RESULTS S3: Sensitivity and replication (homophilic mixing) | FC
RESULTS S4: Sensitivity and replication (homophilic mixing) | animals
RESULTS S5: Partial assortativity
RESULTS S6: Multiple linear regression and dominance analysis
RESULTS S7: Relationship between assortativity and PC1
RESULTS S8: Sensitivity and replication (heterophilic mixing)
RESULTS S9: Homophilic ratios in the functional connectome
RESULTS S10: Sensitivity: Sensitivity and replication (homophilic ratios)

FIGURE 1: Annotated connectomes
FIGURE 3: Standardized assortativity of micro-architectural annotations
FIGURE 4: Assortative mixing of long-range connections
FIGURE 5: Heterophilic mixing
FIGURE 6: Local assortative mixing
FIGURE S1: Assortativity relative to spatially-naive nulls
FIGURE S2: Sensitivity and replication (homophilic mixing) | SC
FIGURE S3: Sensitivity and replication (homophilic mixing) | FC
FIGURE S4: Sensitivity and replication (homophilic mixing) | animals
FIGURE S5: Partial assortativity
FIGURE S6: Multiple linear regression and dominance analysis
FIGURE S7: Relationship between PC1 and assortativity
FIGURE S8: Sensitivity and replication (heterophilic mixing)
FIGURE S9: Homophilic ratios in the functional connectome
FIGURE S10: Sensitivity and replication (homophilic ratios)
FIGURE S11: Homophilic ratios in the SC communities
FIGURE S12: Neurosynth correlations

@author: Vincent Bazinet
"""

import os

# CHANGE THIS TO THE PATH TO THE GIT-HUB REPOSITORY
os.chdir((os.path.expanduser("~") + "/OneDrive - McGill University/"
          "projects (current)/assortativity/git-hub repository/bazinet_assortativity"))

# IMPORT STATEMENTS
import functions as fn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netneurotools.plotting import plot_fsaverage
from netneurotools.datasets import fetch_schaefer2018, fetch_cammoun2012
from tqdm import trange
from scipy.stats import zscore, pearsonr
from statsmodels.stats.multitest import multipletests

# MATPLOTLIB RCPARAMS
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'Calibri'})
plt.rcParams.update({'font.weight': 'light'})
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'svg.fonttype': 'none'})
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

# LOAD COLORMAPS
cmaps = fn.get_colormaps()

# %% DATA: Load data used in the experiments

# Human connectomes
human_SC = fn.load_data("data/human_SC.pickle")
human_FC = fn.load_data("data/human_FC.pickle")
human_annotations = ['receptor_den', 'EI_ratio', 'genePC1', 't1t2', 'thi']

# Human connectomes (supplementary)
human_SC_s400 = fn.load_data('data/human_SC_s400.pickle')
human_SC_L = fn.load_data('data/human_SC_L.pickle')
human_SC_219 = fn.load_data('data/human_SC_219.pickle')
human_SC_1000 = fn.load_data('data/human_SC_1000.pickle')
human_SC_nolog = fn.load_data('data/human_SC_nolog.pickle')
human_FC_s400 = fn.load_data('data/human_FC_s400.pickle')
human_FC_L = fn.load_data('data/human_FC_L.pickle')
human_FC_219 = fn.load_data('data/human_FC_219.pickle')
human_FC_1000 = fn.load_data('data/human_FC_1000.pickle')

# Macaque connectome
scholtens = fn.load_data("data/macaque_scholtens.pickle")
macaque_annotations = ['t1t2', 'thi', 'neuron_den']

# Mouse connectome
oh = fn.load_data("data/mouse_oh.pickle")
mouse_annotations = ['genePC1']

# Receptor densities
receptor_densities = fn.load_data("data/receptor_densities.pickle")
receptor_densities_s400 = fn.load_data("data/receptor_densities_s400.pickle")
receptor_densities_219 = fn.load_data("data/receptor_densities_219.pickle")
receptor_densities_L = fn.load_data("data/receptor_densities_L.pickle")

# Laminar thicknesses
laminar_thicknesses = fn.load_data("data/laminar_thicknesses.pickle")

# Neurosynth
neurosynth = fn.load_data("data/neurosynth.pickle")

# %% RESULTS 3: Standardized assortativity of micro-architectural annotations

result1_path = "results/standardized_assortativity"

# human (structural)
human_SC_results = fn.compute_standardized_assortativity(
    human_SC, 'spin', human_annotations, directed=False)
fn.save_data(human_SC_results, f'{result1_path}/human_SC.pickle')

# human (functional)
human_FC_results = fn.compute_standardized_assortativity(
    human_FC, 'spin', human_annotations, directed=False)
fn.save_data(human_FC_results, f'{result1_path}/human_FC.pickle')

# macaque (scholtens)
scholtens_results = fn.compute_standardized_assortativity(
    scholtens, 'moran', macaque_annotations, directed=True)
fn.save_data(scholtens_results, f'{result1_path}/scholtens.pickle')

# mouse (oh)
oh_results = fn.compute_standardized_assortativity(
    oh, 'moran', mouse_annotations, directed=True)
fn.save_data(oh_results, f'{result1_path}/oh.pickle')

# %% RESULTS 4: Assortative mixing of long-range connections

result2_path = "results/assortativity_thresholded"

percent_kept = np.arange(5, 100, 5)

# human (structural)
human_SC_results = fn.compute_assortativity_thresholded(
    human_SC, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(human_SC_results, f'{result2_path}/human_SC.pickle')

# human (functional)
human_FC_results = fn.compute_assortativity_thresholded(
    human_FC, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(human_FC_results, f'{result2_path}/human_FC.pickle')

# macaque (scholtens)
scholtens_results = fn.compute_assortativity_thresholded(
    scholtens, 'moran', macaque_annotations, percent_kept, directed=True)
fn.save_data(scholtens_results, f'{result2_path}/scholtens.pickle')

# mouse (oh)
oh_results = fn.compute_assortativity_thresholded(
    oh, 'moran', mouse_annotations, percent_kept, directed=True)
fn.save_data(oh_results, f'{result2_path}/oh.pickle')

human_SC_results_2 = fn.load_data(f'{result2_path}/human_SC.pickle')

# %% RESULTS 5: Heterophilic mixing

result3_path = "results/heterophilic_assortativity"

'''
laminar thickness
'''

# Use masks to only compute lower triangle of matrix
layer_mask = np.zeros((6, 6))
layer_mask[np.tril_indices(6)] = 1

# structural connectome
SC_laminar_results = fn.compute_heterophilic_assortativity(
    human_SC, laminar_thicknesses['data'].T, human_SC['spin_nulls'],
    laminar_thicknesses['names'], mask=layer_mask)
fn.save_data(SC_laminar_results, f'{result3_path}/SC_laminar.pickle')

# functional connectome
FC_laminar_results = fn.compute_heterophilic_assortativity(
    human_FC, laminar_thicknesses['data'].T, human_FC['spin_nulls'],
    laminar_thicknesses['names'], mask=layer_mask)
fn.save_data(FC_laminar_results, f'{result3_path}/FC_laminar.pickle')

'''
receptor densities
'''

# Use masks to only compute lower triangle of matrix
receptor_mask = np.zeros((19, 19))
receptor_mask[np.tril_indices(19)] = 1

# structural connectome
SC_receptor_results = fn.compute_heterophilic_assortativity(
    human_SC, receptor_densities['data'].T, human_SC['spin_nulls'],
    receptor_densities['names'], mask=receptor_mask)
fn.save_data(SC_receptor_results, f'{result3_path}/SC_receptor.pickle')

# functional connectome
FC_receptor_results = fn.compute_heterophilic_assortativity(
    human_FC, receptor_densities['data'].T, human_FC['spin_nulls'],
    receptor_densities['names'], mask=receptor_mask)
fn.save_data(FC_receptor_results, f'{result3_path}/FC_receptor.pickle')

# %% RESULTS 6: Homophilic ratios

'''
Homophilic ratios
'''
# homophilic ratios
SC_ratios = fn.compute_homophilic_ratio(human_SC, human_annotations)
SC_ratios['mean'] = np.mean([SC_ratios[ann]
                             for ann in human_annotations], axis=0)
fn.save_data(SC_ratios, "results/local_mixing/SC_ratios.pickle")

'''
Correlations with neurosynth
'''

neurosynth_results = {}
neurosynth_results['terms'] = neurosynth['terms']
neurosynth_results['r'] = np.zeros((123))
neurosynth_results['r_spin'] = np.zeros((123, 10000))
for i in trange(123):
    neurosynth_results['r'][i], _ = pearsonr(
        neurosynth['maps'][:, i], SC_ratios['mean'])
    for j in range(10000):
        spin = human_SC['spin_nulls'][:, j]
        neurosynth_results['r_spin'][i, j], _ = pearsonr(
            neurosynth['maps'][spin, i], SC_ratios['mean'])
neurosynth_results['p'] = np.zeros((123))
for i in range(123):
    neurosynth_results['p'][i] = fn.get_p_value(
        neurosynth_results['r_spin'][i, :], neurosynth_results['r'][i])

'''
Neurosynth categories
'''

# Load information about categories
neurosynth_results['categories_labels'] = [
    'action', 'learning and memory', 'other', 'emotion', 'attention',
    'reasoning and decision making', 'executive/cognitive control',
    'social function', 'perception', 'motivation', 'language']
neurosyn_categories = pd.read_csv(
    "data/neurosynth_categories.csv", header=None
    ).values[:, 1].astype(int)

neurosynth_results['categories'] = neurosyn_categories

# Compute mean of each category
neurosynth_results['r_categories'] = np.array(
    [neurosynth_results['r'][neurosyn_categories == i+1].mean()
     for i in range(11)])

# Compute mean after permuting categories
neurosynth_results['r_categories_perm'] = np.zeros((11, 10000))
for i in range(11):
    for j in range(10000):
        perm = np.random.permutation(neurosyn_categories)
        r_perm_mean = neurosynth_results['r'][perm == i+1].mean()
        neurosynth_results['r_categories_perm'][i, j] = r_perm_mean

# Compute p-values
neurosynth_results['r_categories_p'] = np.zeros((11))
for i in range(11):
    neurosynth_results['r_categories_p'][i] = fn.get_p_value(
        neurosynth_results['r_categories_perm'][i, :],
        neurosynth_results['r_categories'][i])
_, neurosynth_results['r_categories_p_fdr'], _, _ = multipletests(
    neurosynth_results['r_categories_p'], method='fdr_by')

# Save neurosynth results
fn.save_data(neurosynth_results, "results/local_mixing/neurosynth.pickle")

# %% RESULTS S1: Assortativity relative to spatially-naive nulls

result1_path = "results/standardized_assortativity"

results = fn.compute_standardized_assortativity(
    human_SC, 'perm', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_SC_perm.pickle')

results = fn.compute_standardized_assortativity(
    human_FC, 'perm', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_FC_perm.pickle')

# %% RESULTS S2: Sensitivity and replication (homophilic mixing)

'''
Standardized assortativity
'''

result1_path = "results/standardized_assortativity"

# SC_s400
results = fn.compute_standardized_assortativity(
    human_SC_s400, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_SC_s400.pickle')

# SC_L
results = fn.compute_standardized_assortativity(
    human_SC_L, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_SC_L.pickle')

# SC_nolog
results = fn.compute_standardized_assortativity(
    human_SC_nolog, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_SC_nolog.pickle')

# SC_moran
results = fn.compute_standardized_assortativity(
    human_SC, 'moran', human_annotations, directed=False,
    moran_kwargs={'species': 'human', 'hemiid': human_SC['hemiid']})
fn.save_data(results, f'{result1_path}/human_SC_moran.pickle')

# SC_burt
results = fn.compute_standardized_assortativity(
    human_SC, 'burt', human_annotations, directed=False, species='human')
fn.save_data(results, f'{result1_path}/human_SC_burt.pickle')

# SC_219
results = fn.compute_standardized_assortativity(
    human_SC_219, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_SC_219.pickle')

# SC_1000
results = fn.compute_standardized_assortativity(
    human_SC_1000, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_SC_1000.pickle')

# Spearman's
results = fn.compute_spearman_assort(human_SC, 'spin', human_annotations)
fn.save_data(results, f'{result1_path}/human_SC_spearman.pickle')

'''
Thresholded assortativity
'''

result2_path = "results/assortativity_thresholded"

percent_kept = np.arange(5, 100, 5)

# SC_s400
results = fn.compute_assortativity_thresholded(
    human_SC_s400, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_SC_s400.pickle')

# SC_L
results = fn.compute_assortativity_thresholded(
    human_SC_L, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_SC_L.pickle')

# SC_nolog
results = fn.compute_assortativity_thresholded(
    human_SC_nolog, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_SC_nolog.pickle')

# SC_moran
results = fn.compute_assortativity_thresholded(
    human_SC, 'moran', human_annotations, percent_kept, directed=False,
    moran_kwargs={'species': 'human', 'hemiid': human_SC['hemiid']})
fn.save_data(results, f'{result2_path}/human_SC_moran.pickle')

# SC_burt
results = fn.compute_assortativity_thresholded(
    human_SC, 'burt', human_annotations, percent_kept, directed=False,
    species='human')
fn.save_data(results, f'{result2_path}/human_SC_burt.pickle')

# SC_219
results = fn.compute_assortativity_thresholded(
    human_SC_219, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_SC_219.pickle')

# SC_1000
results = fn.compute_assortativity_thresholded(
    human_SC_1000, 'spin', human_annotations, percent_kept, directed=False,
    n_nulls=5000)
fn.save_data(results, f'{result2_path}/human_SC_1000.pickle')

# Spearman's
results = fn.compute_spearman_assort_thresholded(
    human_SC, 'spin', human_annotations, percent_kept)
fn.save_data(results, f'{result2_path}/human_SC_spearman.pickle')

# %% RESULTS S3: Sensitivity and replication (homophilic mixing)

'''
Standardized assortativity
'''

result1_path = "results/standardized_assortativity"

# FC_s400
results = fn.compute_standardized_assortativity(
    human_FC_s400, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_FC_s400.pickle')

# FC_219
results = fn.compute_standardized_assortativity(
    human_FC_219, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_FC_219.pickle')

# FC_L
results = fn.compute_standardized_assortativity(
    human_FC_L, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_FC_L.pickle')

# FC_1000
results = fn.compute_standardized_assortativity(
    human_FC_1000, 'spin', human_annotations, directed=False)
fn.save_data(results, f'{result1_path}/human_FC_1000.pickle')

# FC_moran
results = fn.compute_standardized_assortativity(
    human_FC, 'moran', human_annotations, directed=False,
    moran_kwargs={'species': 'human', 'hemiid': human_FC['hemiid']})
fn.save_data(results, f'{result1_path}/human_FC_moran.pickle')

# FC_burt
results = fn.compute_standardized_assortativity(
    human_FC, 'burt', human_annotations, directed=False, species='human')
fn.save_data(results, f'{result1_path}/human_FC_burt.pickle')

# Spearman's
results = fn.compute_spearman_assort(human_FC, 'spin', human_annotations)
fn.save_data(results, f'{result1_path}/human_FC_spearman.pickle')

'''
Thresholded assortativity
'''

result2_path = "results/assortativity_thresholded"

percent_kept = np.arange(5, 100, 5)

# FC_s400
results = fn.compute_assortativity_thresholded(
    human_FC_s400, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_FC_s400.pickle')

# FC_L
results = fn.compute_assortativity_thresholded(
    human_FC_L, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_FC_L.pickle')

# FC_moran
results = fn.compute_assortativity_thresholded(
    human_FC, 'moran', human_annotations, percent_kept, directed=False,
    moran_kwargs={'species': 'human', 'hemiid': human_FC['hemiid']})
fn.save_data(results, f'{result2_path}/human_FC_moran.pickle')

# FC_burt
results = fn.compute_assortativity_thresholded(
    human_FC, 'burt', human_annotations, percent_kept, directed=False,
    species='human')
fn.save_data(results, f'{result2_path}/human_FC_burt.pickle')

# FC_219
results = fn.compute_assortativity_thresholded(
    human_FC_219, 'spin', human_annotations, percent_kept, directed=False)
fn.save_data(results, f'{result2_path}/human_FC_219.pickle')

# FC_1000
results = fn.compute_assortativity_thresholded(
    human_FC_1000, 'spin', human_annotations, percent_kept, directed=False,
    n_nulls=5000)
fn.save_data(results, f'{result2_path}/human_FC_1000.pickle')

# Spearman's
results = fn.compute_spearman_assort_thresholded(
    human_FC, 'spin', human_annotations, percent_kept)
fn.save_data(results, f'{result2_path}/human_FC_spearman.pickle')

# %% RESULTS S4: Sensitivity and replication (homophilic mixing)

'''
Standardized assortativity
'''

result1_path = "results/standardized_assortativity"

# burt nulls (scholtens)
results = fn.compute_standardized_assortativity(
    scholtens, 'burt', macaque_annotations, directed=True, species='macaque')
fn.save_data(results, f'{result1_path}/scholtens_burt.pickle')

# Spearman's (scholtens)
results = fn.compute_spearman_assort(scholtens, 'moran', macaque_annotations)
fn.save_data(results, f'{result1_path}/scholtens_spearman.pickle')

# burt nulls (oh)
results = fn.compute_standardized_assortativity(
    oh, 'burt', mouse_annotations, directed=True, species='mouse')
fn.save_data(results, f'{result1_path}/oh_burt.pickle')

# Spearman's (oh)
results = fn.compute_spearman_assort(oh, 'moran', mouse_annotations)
fn.save_data(results, f'{result1_path}/oh_spearman.pickle')

'''
Thresholded assortativity
'''

result2_path = "results/assortativity_thresholded"

percent_kept = np.arange(5, 100, 5)

# burt nulls (scholtens)
results = fn.compute_assortativity_thresholded(
    scholtens, 'burt', macaque_annotations, percent_kept, directed=True,
    species='macaque')
fn.save_data(results, f'{result2_path}/scholtens_burt.pickle')

# Spearman's (scholtens)
results = fn.compute_spearman_assort_thresholded(
    scholtens, 'moran', macaque_annotations, percent_kept)
fn.save_data(results, f'{result2_path}/scholtens_spearman.pickle')

# burt nulls (oh)
results = fn.compute_assortativity_thresholded(
    oh, 'burt', mouse_annotations, percent_kept, directed=True,
    species='mouse')
fn.save_data(results, f'{result2_path}/oh_burt.pickle')

# Spearman's (oh)
results = fn.compute_spearman_assort_thresholded(
    oh, 'moran', mouse_annotations, percent_kept)
fn.save_data(results, f'{result2_path}/oh_spearman.pickle')

# %% RESULTS S5: Partial assortativity

# human (structural)
human_SC_partial_assort = fn.compute_partial_assortativity_all(
    human_SC, ['EI_ratio', 'receptor_den', 't1t2', 'thi', 'genePC1'])
fn.save_data(human_SC_partial_assort,
             "results/partial_assortativity/human_SC.pickle")

# human (functional)
human_FC_partial_assort = fn.compute_partial_assortativity_all(
    human_FC, ['EI_ratio', 'receptor_den', 't1t2', 'thi', 'genePC1'])
fn.save_data(human_FC_partial_assort,
             "results/partial_assortativity/human_FC.pickle")

# macaque (scholtens)
scholtens_partial_assort = fn.compute_partial_assortativity_all(
    scholtens, macaque_annotations)
fn.save_data(scholtens_partial_assort,
             "results/partial_assortativity/scholtens.pickle")

# %% RESULTS S6: Multiple linear regression and dominance analysis

'''
Human connectome (structural)
'''

A = human_SC['adj']
permutations = human_SC['spin_nulls'][:, :1000]
dominance_results = {}
for key in human_annotations:

    dominance_results[key] = {}
    _, n_nulls = permutations.shape

    # Dominance for empirical network
    dominance, R2, X_models = fn.weighted_multi_regression_and_dominance(
        A, human_SC, key, human_annotations)
    dominance_results[key]['dominance'] = dominance
    dominance_results[key]['R2'] = R2
    dominance_results['X_models'] = X_models

    # Dominance for surrogate networks
    dominance_results[key]['dominance_spin'] = []
    dominance_results[key]['R2_spin'] = []
    for i in trange(n_nulls):
        perm = permutations[:, i]
        A_perm = A[perm, :][:, perm]
        dominance, R2, _ = fn.weighted_multi_regression_and_dominance(
            A_perm, human_SC, key, human_annotations)
        dominance_results[key]['dominance_spin'].append(dominance)
        dominance_results[key]['R2_spin'].append(R2)

dominance_results = fn.reorganize_dominance_results(
    dominance_results, human_annotations, 1000)

fn.save_data(
    dominance_results, "results/regression_and_dominance/human_SC.pickle")

'''
Human connectome (functional)
'''

A = human_FC['adj']
permutations = human_FC['spin_nulls'][:, :1000]
dominance_results = {}
for key in human_annotations:

    dominance_results[key] = {}
    _, n_nulls = permutations.shape

    # Dominance for empirical network
    dominance, R2, X_models = fn.weighted_multi_regression_and_dominance(
        A, human_FC, key, human_annotations)
    dominance_results[key]['dominance'] = dominance
    dominance_results[key]['R2'] = R2
    dominance_results['X_models'] = X_models

    # Dominance for surrogate networks
    dominance_results[key]['dominance_spin'] = []
    dominance_results[key]['R2_spin'] = []
    for i in trange(n_nulls):
        perm = permutations[:, i]
        A_perm = A[perm, :][:, perm]
        dominance, R2, _ = fn.weighted_multi_regression_and_dominance(
            A_perm, human_FC, key, human_annotations)
        dominance_results[key]['dominance_spin'].append(dominance)
        dominance_results[key]['R2_spin'].append(R2)

dominance_results = fn.reorganize_dominance_results(
    dominance_results, human_annotations, 1000)

fn.save_data(
    dominance_results, "results/regression_and_dominance/human_FC.pickle")

'''
macaque connectome (scholtens)
'''

n_nulls = 1000
A = scholtens['adj']
dominance_results = {}
for key in macaque_annotations:

    dominance_results[key] = {}

    # Generate Moran nulls
    permutations = np.zeros((n_nulls, 39))
    nulls = fn.generate_moran_nulls(scholtens[key], scholtens['dist'], n_nulls)
    sorted_ids = np.argsort(scholtens[key])
    for i in range(n_nulls):
        ii = np.argsort(nulls[i, :])
        np.put(permutations[i, :], ii, sorted_ids)

    # Dominance for empirical network
    dominance, R2, X_models = fn.weighted_multi_regression_and_dominance(
        A, scholtens, key, macaque_annotations)
    dominance_results[key]['dominance'] = dominance
    dominance_results[key]['R2'] = R2
    dominance_results['X_models'] = X_models

    # Dominance for surrogate networks
    dominance_results[key]['dominance_spin'] = []
    dominance_results[key]['R2_spin'] = []
    for i in trange(n_nulls):
        perm = permutations[i, :].astype(int)
        A_perm = A[perm, :][:, perm]
        dominance, R2, _ = fn.weighted_multi_regression_and_dominance(
            A_perm, scholtens, key, macaque_annotations)
        dominance_results[key]['dominance_spin'].append(dominance)
        dominance_results[key]['R2_spin'].append(R2)

dominance_results = fn.reorganize_dominance_results(
    dominance_results, macaque_annotations, 1000)

fn.save_data(
    dominance_results, "results/regression_and_dominance/scholtens.pickle")

# %% RESULTS S7: Relationship between assortativity and PC1

PC1_results = {}

# Compute first component
PC1_results['PC1'] = fn.get_PC1(human_FC['adj'])

# Compute correlations between receptor density maps and PC
PC1_results['r_rec'], PC1_results['r_prod_rec'] = fn.correlate_with_PC(
    receptor_densities['data'], -PC1_results['PC1'])

# Compute correlations between laminar thickness maps and PC
PC1_results['r_lay'], PC1_results['r_prod_lay'] = fn.correlate_with_PC(
    laminar_thicknesses['data'], -PC1_results['PC1'])

PC1_results['layers'] = laminar_thicknesses['names']
PC1_results['receptors'] = receptor_densities['names']

fn.save_data(PC1_results,
             'results/PC1_assortativity_relationship/PC1_results.pickle')

# %% RESULTS S8: Sensitivity and replication (heterophilic mixing)

connectomes = [human_SC_s400, human_SC_L, human_SC_219,
               human_FC_s400, human_FC_L, human_FC_219]
names = ["SC_s400", "SC_L", "SC_219", "FC_400", "FC_L", "FC_219"]
densities = [
    receptor_densities_s400, receptor_densities_L, receptor_densities_219,
    receptor_densities_s400, receptor_densities_L, receptor_densities_219]

# Use masks to only compute lower triangle of matrix
receptor_mask = np.zeros((19, 19))
receptor_mask[np.tril_indices(19)] = 1

for connectome, name, density_data in zip(connectomes, names, densities):

    results = fn.compute_heterophilic_assortativity(
        connectome, receptor_densities['data'].T, connectome['spin_nulls'],
        receptor_densities['names'], mask=receptor_mask)
    fn.save_data(results, f'{result3_path}/{name}_receptor.pickle')

# %% RESULTS S9: Homophilic ratios in the functional connectome

FC_ratios = fn.compute_homophilic_ratio(human_FC, human_annotations)
FC_ratios['mean'] = np.mean([FC_ratios[ann]
                             for ann in human_annotations], axis=0)
fn.save_data(FC_ratios, "results/local_mixing/FC_ratios.pickle")

# %% RESULTS S10: Sensitivity and replication (homophilic ratios)

supplementary_connectomes = {'SC_s400': human_SC_s400,
                             'SC_L': human_SC_L,
                             'SC_219': human_SC_219,
                             'SC_1000': human_SC_1000}

for label, connectome in supplementary_connectomes.items():
    ratios = fn.compute_homophilic_ratio(connectome, human_annotations)
    ratios['mean'] = np.mean([ratios[ann]
                             for ann in human_annotations], axis=0)
    fn.save_data(ratios, f"results/local_mixing/{label}_ratios.pickle")

# %% FIGURE 1: Annotated connectomes

'''
human (structural) connectome
'''

# threshold connectome (for visualization)
adj = human_SC['adj'].copy()
adj[adj < np.percentile(adj, 98)] = 0

# plot connectome
fn.plot_network(
    adj, human_SC['coords'][:, :2], adj, None, edge_alpha=0.5, s=15,
    figsize=(5, 5), node_cmap="Greys", edge_cmap=cmaps['Reds_3'],
    linewidth=0.5)
save_path = "figures/figure 1/human (structural).png"
plt.savefig(save_path, dpi=600, transparent=True)

'''
human (functional) connectome
'''

# threshold connectome (for visualization)
adj = human_FC['adj'].copy()
adj[adj < np.percentile(adj, 98.5)] = 0

# plot connectome
fn.plot_network(
    adj, human_FC['coords'][:, :2], adj, None, edge_alpha=0.5, s=15,
    figsize=(5, 5), node_cmap="Greys", edge_cmap=cmaps['Blues_3'],
    linewidth=0.5)
save_path = "figures/figure 1/human (functional).png"
plt.savefig(save_path, dpi=600, transparent=True)

'''
human annotations
'''

for ann in human_annotations:
    scores = human_SC[ann]
    node_size = 15 * (3 + zscore(scores) + abs(zscore(scores).min()))
    fn.plot_network(
        None, human_SC['coords'][:, :2], None, scores, s=node_size,
        node_cmap=cmaps['Spectral_11_r'], view_edge=False, edge_alpha=0.10)
    save_path = f"figures/figure 1/human - {ann}.png"
    plt.savefig(save_path, dpi=600, transparent=True)

'''
macaque (scholtens) connectome
'''

# bilaterize network (for visualization)
adj, coords = fn.bilaterize_network(
    scholtens['adj'], scholtens['coords'], symmetry_axis=0)

# plot connectome
fn.plot_network(
    adj, coords[:, :2], adj, None, linewidth=1.5, edge_alpha=0.1,
    s=100, figsize=(5, 5), node_cmap="Greys", edge_cmap="Greens")
plt.savefig("figures/figure 1/macaque (scholtens).png", dpi=600,
            transparent=True)

'''
macaque (scholtens) annotations
'''

for ann in macaque_annotations:

    # bilaterize annotation scores (for visualization)
    scores = np.concatenate((scholtens[ann], scholtens[ann]))

    fn.plot_network(
        None, coords[:, :2], None, scores, node_cmap=cmaps['Spectral_11_r'],
        s=500, view_edge=False)
    plt.savefig(f"figures/figure 1/macaque - {ann}.png",
                transparent=True, dpi=600)

'''
mouse (oh) connectome
'''

# bilaterize network (for visualization)
adj, coords = fn.bilaterize_network(
    oh['adj'], oh['coords'], symmetry_axis=2,
    between_hemi_dist=-oh['coords'][:, 2].max() + 100)

# threshold connectome and binarize (for visualization)
adj[adj < np.percentile(oh['adj'], 80)] = 0
adj[adj > 0] = 1

# plot connectome
fn.plot_network(
    adj, coords[:, [0, 2]], None, None, linewidth=1.5, edge_alpha=0.01, s=50,
    figsize=(5, 5), node_cmap="Greys", edge_vmin=0, edges_color="#3f007d64")
plt.savefig("figures/figure 1/mouse (oh).png",
            dpi=600, transparent=True)

'''
mouse (oh) annotations
'''

for ann in mouse_annotations:

    # bilaterize annotation scores (for visualization)
    scores = np.concatenate((oh[ann], oh[ann]))

    fn.plot_network(
        None, coords[:, [0, 2]], None, scores, linewidth=1.5, edge_alpha=0.01,
        s=50, figsize=(5, 5), view_edge=False,
        node_cmap=cmaps['Spectral_11_r'])
    plt.savefig(f"figures/figure 1/mouse - {ann}.png",
                transparent=True, dpi=600)

# %% FIGURE 3: Standardized assortativity of micro-architectural annotations

'''
Panel (a): assortativity compared to spatial autocorrelation-preserving nulls
'''

# human (structural)
fig = fn.assortativity_boxplot(
    "human_SC", 'spin', human_annotations, face_color='#FCBBA1',
    edge_color='#CB181D', figsize=(2.5, 2))
fig.savefig("figures/figure 3/boxplot_human_SC.svg")

# human (functional)
fig = fn.assortativity_boxplot(
    "human_FC", 'spin', human_annotations, face_color='#C6DBEF',
    edge_color='#2171B5', figsize=(2.5, 2))
fig.savefig("figures/figure 3/boxplot_human_FC.svg")

# macaque (scholtens)
fig = fn.assortativity_boxplot(
    "scholtens", 'moran', macaque_annotations, face_color='#C7E9C0',
    edge_color='#238B45', figsize=(1.5, 2))
fig.savefig("figures/figure 3/boxplot_scholtens.svg")

# mouse (oh)
fig = fn.assortativity_boxplot(
    "oh", 'moran', mouse_annotations, face_color='#DADAEB',
    edge_color='#6A51A3', figsize=(0.5, 2))
fig.savefig("figures/figure 3/boxplot_oh.svg")

'''
panel (b): barplots with standardized assortativity results
'''

path_fig3_results = "results/standardized_assortativity"
assort_results = [fn.load_data(f"{path_fig3_results}/human_SC.pickle")[ann]
                  for ann in human_annotations] + \
                 [fn.load_data(f"{path_fig3_results}/human_FC.pickle")[ann]
                  for ann in human_annotations] + \
                 [fn.load_data(f"{path_fig3_results}/scholtens.pickle")[ann]
                  for ann in macaque_annotations] + \
                 [fn.load_data(f"{path_fig3_results}/oh.pickle")[ann]
                  for ann in mouse_annotations]

labels = ['receptor den.', 'E/I ratio', 'gene PC1', 'T1w/T2w', 'thickness',
          'receptor den.', 'E/I ratio', 'gene PC1', 'T1w/T2w', 'thickness',
          'T1w/T2w', 'thickness', 'neuron/cell ratio', 'genePC1']
sig_colors = (['#CB181D'] * 5) + (['#2171B5'] * 5) + \
             (['#238B45'] * 3) + (['#6A51A3'] * 1)
non_sig_colors = (['#FCBBA1'] * 5) + (['#C6DBEF'] * 5) + \
                 (['#C7E9C0'] * 3) + (['#DADAEB'] * 1)
null_types = (['spin'] * 5) + (['spin'] * 5) + \
             (['moran'] * 3) + (['moran'] * 1)

fig = fn.assortativity_barplot(assort_results, labels, non_sig_colors,
                               sig_colors, null_types)
fig.savefig("figures/figure 3/barplot.svg")

# %% FIGURE 4: Assortative mixing of long-range connections

percent_kept = np.arange(5, 100, 5)

# human (structural)
fig = fn.plot_assortativity_thresholded(
    "human_SC", human_annotations, percent_kept,
    ['#28c828ff', '#1a1ad6ff', '#f2b701ff', '#c828c6ff', '#1ad6d4ff'],
    ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff'])
fig.savefig("figures/figure 4/lineplot_human_SC.svg")

# human (functional)
fig = fn.plot_assortativity_thresholded(
    "human_FC", human_annotations, percent_kept,
    ['#28c828ff', '#1a1ad6ff', '#f2b701ff', '#c828c6ff', '#1ad6d4ff'],
    ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff'])
fig.savefig("figures/figure 4/lineplot_human_FC.svg")

# macaque (scholtens)
fig = fn.plot_assortativity_thresholded(
    "scholtens", macaque_annotations, percent_kept,
    ['#c828c6ff', '#1ad6d4ff', '#d61a1aff'],
    ['#f5cdf5ff', '#caf8f8ff', '#f8cacaff'])
fig.savefig("figures/figure 4/lineplot_macaque.svg")

# mouse (oh)
fig = fn.plot_assortativity_thresholded(
    "oh", mouse_annotations, percent_kept,
    ['#f2b701ff'], ['#feeebaff'])
fig.savefig("figures/figure 4/lineplot_mouse.svg")

# %% FIGURE 5: Heterophilic mixing

path_results_fig5 = "results/heterophilic_assortativity"

'''
Identify min and max assortativity values across network + annotation types
'''

m = 0
for matrix_name in ['SC_laminar', 'FC_laminar', 'SC_receptor', 'FC_receptor']:
    matrix = fn.load_data(f'{path_results_fig5}/{matrix_name}.pickle')
    max_assort = np.nanmax(np.abs(matrix['a_z']))
    if max_assort > m:
        m = max_assort

'''
panel (b): mixing matrices (laminar thickness)
'''

layer_names = laminar_thicknesses['names']

# structural connectome
SC_laminar = fn.load_data(f'{path_results_fig5}/SC_laminar.pickle')
assort_z = fn.fill_triu(SC_laminar['a_z'])
assort_p_fdr = fn.fill_triu(SC_laminar['a_p_fdr'])
fn.plot_heatmap(assort_z, layer_names, layer_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=2, figsize=(3.4, 3.4),
                sigs=assort_p_fdr < 0.05, text_size=17)
plt.savefig("figures/figure 5/heterophilic_matrix_SC_laminar.svg")

# functional connectome
FC_laminar = fn.load_data(f'{path_results_fig5}/FC_laminar.pickle')
assort_z = fn.fill_triu(FC_laminar['a_z'])
assort_p_fdr = fn.fill_triu(FC_laminar['a_p_fdr'])
fn.plot_heatmap(assort_z, layer_names, layer_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=2, figsize=(3.4, 3.4),
                sigs=assort_p_fdr < 0.05, text_size=17)
plt.savefig("figures/figure 5/heterophilic_matrix_FC_laminar.svg")

'''
panel (b): mixing matrices (receptor densities)
'''

receptor_names = receptor_densities['names']

# structural connectome
SC_receptor = fn.load_data(f'{path_results_fig5}/SC_receptor.pickle')
assort_z = fn.fill_triu(SC_receptor['a_z'])
assort_p_fdr = fn.fill_triu(SC_receptor['a_p_fdr'])
fn.plot_heatmap(assort_z, receptor_names, receptor_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=1, figsize=(5, 5),
                sigs=assort_p_fdr < 0.05)
plt.savefig("figures/figure 5/heterophilic_matrix_SC_receptors.svg")

# functional connectome
FC_receptor = fn.load_data(f'{path_results_fig5}/FC_receptor.pickle')
assort_z = fn.fill_triu(FC_receptor['a_z'])
assort_p_fdr = fn.fill_triu(FC_receptor['a_p_fdr'])
fn.plot_heatmap(assort_z, receptor_names, receptor_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=1, figsize=(5, 5),
                sigs=assort_p_fdr < 0.05)
plt.savefig("figures/figure 5/heterophilic_matrix_FC_receptors.svg")

'''
panel (c): scatterplot of assortativity results for SC and FC
'''

fn.plot_SC_FC_heterophilic_comparison(
    SC_receptor, FC_receptor, SC_laminar, FC_laminar)
plt.savefig("figures/figure 5/scatterplot_SC_FC.svg")

# %% FIGURE 6: Local assortative mixing

'''
load results
'''

SC_ratios = fn.load_data("results/local_mixing/SC_ratios.pickle")

'''
panel (c): homophilic ratios
'''

# Find maximal + minimal ratios (for colormap) | set at 97.5 percentile
ratios_all = [SC_ratios[ann] for ann in human_annotations]
ratios_min = np.percentile(ratios_all, 2.5)
ratios_max = np.percentile(ratios_all, 97.5)

# plot homophilic ratios on brain surface
for ann in human_annotations:

    SC_ratio = SC_ratios[ann]

    lhannot, rhannot = fetch_schaefer2018()['800Parcels7Networks']
    surface_image, dot_image = fn.plot_homophilic_ratios(
        SC_ratio, ann, human_SC['coords'], lhannot=lhannot, rhannot=rhannot,
        noplot=human_SC['noplot'], order=human_SC['order'], vmin=ratios_min,
        vmax=ratios_max)

    surface_image.save_image(
        f"figures/figure 6/homophilic_ratio_surface_{ann}.png", mode='rgba')

    dot_image.savefig(f"figures/figure 6/homophilic_ratio_dot_{ann}.png",
                      dpi=300, transparent=True)

'''
panel (d): topology and geometry
'''

strength = human_SC['adj'].sum(axis=0)
mcd = np.average(human_SC['dist'], weights=human_SC['adj'], axis=0)

# Node strength
fig, r_results = fn.plot_regression(
    strength, SC_ratios['mean'], x_label="node strength",
    y_label='homophilic ratio (average)', permutations=human_SC['spin_nulls'])
fig.savefig("figures/figure 6/scatterplot_str.svg", dpi=300)

# Mean connection length
fig, r_results = fn.plot_regression(
    mcd, SC_ratios['mean'], x_label="mean connection distance",
    y_label='homophilic ratio (average)', permutations=human_SC['spin_nulls'])
plt.savefig("figures/figure 6/scatterplot_mcd.svg", dpi=300)

'''
panel (e): neurosynth
'''

neurosynth_results = fn.load_data("results/local_mixing/neurosynth.pickle")

r_categories = neurosynth_results['r_categories']
pfdr_categories = neurosynth_results['r_categories_p_fdr']
categories_labels = neurosynth_results['categories_labels']
order = np.argsort(r_categories)
plt.figure(figsize=(6, 3))
colors = np.array(np.arange(11), dtype='str')
colors[r_categories < 0] = '#f7fcf0ff'
colors[r_categories > 0] = '#084081ff'
colors[pfdr_categories > 0.05] = 'white'
plt.bar(np.arange(11),
        r_categories[order],
        color=colors[order],
        edgecolor='black')
plt.xticks(np.arange(11),
           np.array(categories_labels)[order],
           rotation='vertical')
plt.savefig("figures/figure 6/barplot_neurosynth_categories.svg")

# Retrieve some important data
categories_labels = neurosynth_results['categories_labels']
r_all = neurosynth_results['r']
categories = neurosynth_results['categories']

# Order categories based on correlations
order = np.argsort(neurosynth_results['r_categories'])
categories_ordered = np.array(neurosynth_results['categories_labels'])[order]

# Set colors of the boxplots
colors = np.array(np.arange(11), dtype='str')
colors[neurosynth_results['r_categories'][order] < 0] = '#f7fcf0ff'
colors[neurosynth_results['r_categories'][order] > 0] = '#b2d4fbff'
colors[neurosynth_results['r_categories_p_fdr'][order] > 0.05] = 'white'

# Plot the figure
plt.figure(figsize=(4, 2))
flierprops = dict(marker='+', markerfacecolor='lightgray',
                  markeredgecolor='lightgray')
for i, label in enumerate(categories_ordered):
    category_id = np.where(np.array(categories_labels) == label)[0][0]
    r_categories_all = r_all[categories == category_id+1]

    # Add jitter
    plt.scatter(np.random.normal(i+1, 0.1, size=len(r_categories_all)),
                r_categories_all,
                edgecolors='gray',
                facecolors="white",
                rasterized=True,
                s=15, zorder=1)

    # Overlay boxplot on top of scatter points
    bplot = plt.gca().boxplot(
        r_categories_all, positions=[i+1], widths=0.8, patch_artist=True,
        showfliers=False, showcaps=False, zorder=0,
        medianprops=dict(color='black'))
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[i])

plt.xticks(np.arange(11), categories_ordered, rotation='vertical')
plt.savefig("figures/figure 6/boxplot_neurosynth_categories.svg", dpi=600)

# %% FIGURE S1: Assortativity relative to spatially-naive nulls

fig = fn.assortativity_boxplot(
    "human_SC_perm", 'perm', human_annotations, face_color='#FCBBA1',
    edge_color='#CB181D', figsize=(2.5, 2))
fig.savefig("figures/figure s1/boxplot_human_SC_perm.svg", bbox_inches='tight')

fig = fn.assortativity_boxplot(
    "human_FC_perm", 'perm', human_annotations, face_color='#C6DBEF',
    edge_color='#2171B5', figsize=(2.5, 2))
fig.savefig("figures/figure s1/boxplot_human_FC_perm.svg", bbox_inches='tight')

# %% FIGURE S2: Sensitivity and replication (homophilic mixing) | SC

networks = ['SC_s400', 'SC_L', 'SC_219', 'SC_1000', 'SC_nolog',
            'SC_moran', 'SC_burt', 'SC_spearman']

'''
Standardized assortativity
'''

# Fetch results for all networks (to get the y-lims for the figures)
results_all = []
for network in networks:
    results = fn.load_data(f"results/standardized_assortativity/human_{network}.pickle")
    w_ga_z = [results[ann]['assort_z'] for ann in human_annotations]
    results_all.append(w_ga_z)
results_all = np.array(results_all)

# Plot the barplots for each network
for name in networks:

    # Get relevant information
    results = [fn.load_data(f"results/standardized_assortativity/human_{name}.pickle")[ann]
               for ann in human_annotations]
    colors = ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff']
    sig_colors = ['#28a228ff', '#264dbdff', '#f2b701ff', '#c349c1ff', '#2cbebeff']

    fn.assortativity_barplot(results, human_annotations, colors, sig_colors,
                             figsize=(2.1, 1.5), barwidth=0.65,
                             ylim=(results_all.min(), results_all.max()))

    plt.savefig(f"figures/figure s2/z_assort_{name}.svg", bbox_inches='tight')

'''
assortativity thresholded
'''

percent_kept = np.arange(5, 100, 5)
networks = ['human_SC_s400', 'human_SC_L', 'human_SC_219', 'human_SC_1000',
            'human_SC_nolog', 'human_SC_moran', 'human_SC_burt',
            'human_SC_spearman']

for name in networks:
    fig = fn.plot_assortativity_thresholded(
        name, human_annotations, percent_kept,
        ['#28c828ff', '#1a1ad6ff', '#f2b701ff', '#c828c6ff', '#1ad6d4ff'],
        ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff'],
        figsize=(2.4, 1.5))
    fig.savefig(f"figures/figure s2/thres_assort_{name}.svg", bbox_inches='tight')

# %% FIGURE S3: Sensitivity and replication (homophilic mixing) | FC

'''
Standardized assortativity
'''

networks = ['FC_s400', 'FC_L', 'FC_219', 'FC_1000', 'FC_moran', 'FC_burt',
            'FC_spearman']

# Fetch results for all networks (to get the y-lims for the figures)
results_all = []
for network in networks:
    results = fn.load_data(f"results/standardized_assortativity/human_{network}.pickle")
    w_ga_z = [results[ann]['assort_z'] for ann in human_annotations]
    results_all.append(w_ga_z)
results_all = np.array(results_all)

# Plot the barplots for each network
for name in networks:

    # Get relevant information
    results = [fn.load_data(f"results/standardized_assortativity/human_{name}.pickle")[ann]
               for ann in human_annotations]
    colors = ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff']
    sig_colors = ['#28a228ff', '#264dbdff', '#f2b701ff', '#c349c1ff', '#2cbebeff']

    fn.assortativity_barplot(results, human_annotations, colors, sig_colors,
                             figsize=(2.1, 1.5), barwidth=0.65,
                             ylim=(results_all.min(), results_all.max()))

    plt.savefig(f"figures/figure s3/z_assort_{name}.svg", bbox_inches='tight')

'''
assortativity thresholded
'''

percent_kept = np.arange(5, 100, 5)
networks = ['human_FC_s400', 'human_FC_L', 'human_FC_219', 'human_FC_1000',
            'human_FC_moran', 'human_FC_burt', 'human_FC_spearman']

for name in networks:
    fig = fn.plot_assortativity_thresholded(
        name, human_annotations, percent_kept,
        ['#28c828ff', '#1a1ad6ff', '#f2b701ff', '#c828c6ff', '#1ad6d4ff'],
        ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff'],
        figsize=(2.4, 1.5))
    fig.savefig(f"figures/figure s3/thres_assort_{name}.svg", bbox_inches='tight')

# %% FIGURE S4: Sensitivity and replication (homophilic mixing) | animals

'''
Standardized assortativity | panel (a): macaque connectome
'''

networks = ['scholtens_burt', 'scholtens_spearman']

# Fetch results for all networks (to get the y-lims for the figures)
results_all = []
for network in networks:
    results = fn.load_data(f"results/standardized_assortativity/{network}.pickle")
    w_ga_z = [results[ann]['assort_z'] for ann in macaque_annotations]
    results_all.append(w_ga_z)
results_all = np.array(results_all)

# Plot the barplots for each network
for name in networks:

    # Get relevant information
    results = [fn.load_data(f"results/standardized_assortativity/{name}.pickle")[ann]
               for ann in macaque_annotations]
    colors = ['#f5cdf5ff', '#caf8f8ff', '#f8cacaff']
    sig_colors = ['#c349c1ff', '#2cbebeff', '#d61a1aff']

    fn.assortativity_barplot(results, macaque_annotations, colors, sig_colors,
                             figsize=(1.5, 1.5), barwidth=0.65,
                             ylim=(results_all.min(), results_all.max()),
                             tight_layout=False)

    plt.savefig(f"figures/figure s4/z_assort_{name}.svg", bbox_inches='tight')

'''
Standardized assortativity | panel (b): mouse connectome
'''

networks = ['oh_burt', 'oh_spearman']

# Fetch results for all networks (to get the y-lims for the figures)
results_all = []
for network in networks:
    results = fn.load_data(f"results/standardized_assortativity/{network}.pickle")
    w_ga_z = [results[ann]['assort_z'] for ann in mouse_annotations]
    results_all.append(w_ga_z)
results_all = np.array(results_all)

# Plot the barplots for each network
for name in networks:

    # Get relevant information
    results = [fn.load_data(f"results/standardized_assortativity/{name}.pickle")[ann]
               for ann in mouse_annotations]
    colors = ['#feeebaff']
    sig_colors = ['#f2b701ff']

    fn.assortativity_barplot(results, mouse_annotations, colors, sig_colors,
                             figsize=(0.5, 1.5), barwidth=0.65,
                             ylim=(results_all.min(), results_all.max()),
                             tight_layout=False)

    plt.savefig(f"figures/figure s4/z_assort_{name}.svg",
                bbox_inches='tight')

'''
Thresholded assortativity | panels (a)
'''

percent_kept = np.arange(5, 100, 5)

networks = ['scholtens_burt', 'scholtens_spearman']

for name in networks:
    fig = fn.plot_assortativity_thresholded(
        name, macaque_annotations, percent_kept,
        ['#c828c6ff', '#1ad6d4ff', '#d61a1aff'],
        ['#f5cdf5ff', '#caf8f8ff', '#f8cacaff'],
        figsize=(2.4, 1.5))
    fig.savefig(f"figures/figure s4/thres_assort_{name}.svg",
                bbox_inches='tight')

temp = fn.load_data("results/assortativity_thresholded/scholtens_spearman.pickle")

'''
Thresholded assortativity | panels (b)
'''

percent_kept = np.arange(5, 100, 5)

networks = ['oh_burt', 'oh_spearman']

for name in networks:
    fig = fn.plot_assortativity_thresholded(
        name, mouse_annotations, percent_kept,
        ['#f2b701ff'],
        ['#feeebaff'],
        figsize=(2.4, 1.5))
    fig.savefig(f"figures/figure s4/thres_assort_{name}.svg",
                bbox_inches='tight')


# %% FIGURE S5: Partial assortativity

'''
panel (a): human (structural)
'''

results = fn.load_data("results/partial_assortativity/human_SC.pickle")
n_keys = len(results['keys'])
results['r'][np.diag_indices(n_keys)] = np.nan
m = np.nanmax(results['r'])
fn.plot_heatmap(results['r'], results['keys'], results['keys'],
                text=True, cmap='Reds', vmin=0, vmax=m,
                grid_width=2, figsize=(3.1, 3.1), text_size=11,
                tight_layout=False)
plt.savefig("figures/figure s5/human_SC_heatmap.svg")

'''
panel (b): human (functional)
'''

results = fn.load_data("results/partial_assortativity/human_FC.pickle")
n_keys = len(results['keys'])
results['r'][np.diag_indices(n_keys)] = np.nan
m = np.nanmax(results['r'])
fn.plot_heatmap(results['r'], results['keys'], results['keys'],
                text=True, cmap='Blues', vmin=0, vmax=m,
                grid_width=2, figsize=(3.1, 3.1), text_size=11,
                tight_layout=False)
plt.savefig("figures/figure s5/human_FC_heatmap.svg")

'''
panel (c): mouse (scholtens)
'''

results = fn.load_data("results/partial_assortativity/scholtens.pickle")
n_keys = len(results['keys'])
results['r'][np.diag_indices(n_keys)] = np.nan
m = np.nanmax(results['r'])
fn.plot_heatmap(results['r'], results['keys'], results['keys'],
                text=True, cmap='Greens', vmin=0, vmax=m,
                grid_width=2, figsize=(1.86, 1.86), text_size=11,
                tight_layout=False)
plt.savefig("figures/figure s5/scholtens_heatmap.svg")

# %% FIGURE S6: Multiple linear regression and dominance analysis

results_path = "results/regression_and_dominance"

'''
human (SC)
'''

dominance_results = fn.load_data(f"{results_path}/human_SC.pickle")

fn.boxplot(np.flip(dominance_results['R2_spin']), vert=False,
           edge_colors='#cb181dff', face_colors='#fcbba1ff',
           figsize=(2.5, 2.5))
plt.scatter(np.flip(dominance_results['R2']), np.arange(1, 6), c='#cb181dff')
plt.xlabel("R2")
plt.yticks(np.arange(1, 6), np.flip(np.asarray(dominance_results['keys'])))
plt.savefig("figures/figure s6/human_SC_multiple_regression.svg")

fn.plot_heatmap(dominance_results['dominance_percentage'],
                dominance_results['keys'],
                dominance_results['keys'],
                text=True, cmap='Reds', grid_width=2, figsize=(3.1, 3.1),
                text_size=11, tight_layout=False)
plt.savefig("figures/figure s6/human_SC_dominance.svg", dpi=600)

'''
human (FC)
'''

dominance_results = fn.load_data(f"{results_path}/human_FC.pickle")

fn.boxplot(np.flip(dominance_results['R2_spin']), vert=False,
           edge_colors='#2171b5ff', face_colors='#c6dbefff',
           figsize=(2.5, 2.5))
plt.scatter(np.flip(dominance_results['R2']), np.arange(1, 6), c='#2171b5ff')
plt.xlabel("R2")
plt.yticks(np.arange(1, 6), np.flip(np.asarray(dominance_results['keys'])))
plt.savefig("figures/figure s6/human_FC_multiple_regression.svg")

fn.plot_heatmap(dominance_results['dominance_percentage'],
                dominance_results['keys'],
                dominance_results['keys'],
                text=True, cmap='Blues', grid_width=2, figsize=(3.1, 3.1),
                text_size=11, tight_layout=False)
plt.savefig("figures/figure s6/human_FC_dominance.svg", dpi=600)

'''
scholtens
'''

dominance_results = fn.load_data(f"{results_path}/scholtens.pickle")

fn.boxplot(np.flip(dominance_results['R2_spin']), vert=False,
           edge_colors='#c7e9c0ff', face_colors='#238b45ff',
           figsize=(1.5, 1.5))
plt.scatter(np.flip(dominance_results['R2']), np.arange(1, 4), c='#238b45ff')
plt.xlabel("R2")
plt.yticks(np.arange(1, 4), np.flip(np.asarray(dominance_results['keys'])))
plt.savefig("figures/figure s6/scholtens_multiple_regression.svg")

fn.plot_heatmap(dominance_results['dominance_percentage'],
                dominance_results['keys'],
                dominance_results['keys'],
                text=True, cmap='Greens', grid_width=2, figsize=(1.86, 1.86),
                text_size=11, tight_layout=False)
plt.savefig("figures/figure s6/scholtens_dominance.svg", dpi=600)

# %% FIGURE S7: Relationship between PC1 and assortativity

PC1_results = fn.load_data(
    'results/PC1_assortativity_relationship/PC1_results.pickle')

hetero_mix_FC_laminar = fn.load_data(
    "results/heterophilic_assortativity/FC_laminar.pickle")

hetero_mix_FC_receptors = fn.load_data(
    "results/heterophilic_assortativity/FC_receptor.pickle")

'''
panel (a): FC PC1
'''

# plot homophilic ratios on brain surface
lhannot, rhannot = fetch_schaefer2018()['800Parcels7Networks']
im = plot_fsaverage(
    -PC1_results['PC1'], lhannot=lhannot, rhannot=rhannot,
    noplot=human_SC['noplot'], order=human_SC['order'], views=['lateral', 'm'],
    colormap=cmaps['Spectral_11_r'],
    data_kws={'representation': 'wireframe', 'line_width': 4.0},
    vmin=(-PC1_results['PC1']).min(), vmax=(-PC1_results['PC1']).max())
im.save_image("figures/figure s7/FC_PC1.png", mode='rgba')

# plot homophilic ratios on dotted brain
scores = -PC1_results['PC1']
size_change = abs(zscore(scores))
size_change[size_change > 5] = 5
size = 65 + (10 * size_change)
fn.plot_network(None, human_SC['coords'][:, :2], None, scores,
                s=size, view_edge=False, node_cmap=cmaps['Spectral_11_r'])
plt.savefig("figures/figure s7/FC_PC1_dots.png", dpi=600, transparent=True)

'''
panel (b): relationship with attribute
'''

# Adjacency matrix, ordered along gradient
A = human_FC['adj'].copy()
PC1 = PC1_results['PC1']
A = A[np.argsort(PC1), :][:, np.argsort(PC1)]

plt.figure()
plt.imshow(A, cmap=cmaps['SunsetDark_7'])
plt.savefig("figures/figure s7/adj_PC1_ordered.svg", dpi=600)

'''
panel (c): FC PC1 and z-assortativity (layers)
'''

fig1, fig2, fig3, reg = fn.plot_PC1_assortativity_correlations(
    PC1_results['r_lay'], PC1_results['r_prod_lay'], laminar_thicknesses,
    hetero_mix_FC_laminar['a_z'], barplot_size=(2,2), grid_width=2)
fig1.savefig("figures/figure s7/correlation_PC1_layers.svg")
fig2.savefig("figures/figure s7/correlation_zassort_rprod_layers.svg", dpi=300)
fig3.savefig("figures/figure s7/heatmap_rprod_layers.svg", dpi=300)

'''
panel (d): FC PC1 and z-assortativity (receptors)
'''

fig1, fig2, fig3, reg = fn.plot_PC1_assortativity_correlations(
    PC1_results['r_rec'], PC1_results['r_prod_rec'], receptor_densities,
    hetero_mix_FC_receptors['a_z'], barplot_size=(5, 2), grid_width=0.25)
fig1.savefig("figures/figure s7/correlation_PC1_receptors.svg")
fig2.savefig("figures/figure s7/correlation_zassort_rprod_receptors.svg",
             dpi=300)
fig3.savefig("figures/figure s7/heatmap_rprod_receptors.svg",
             dpi=300)

# %% FIGURE S8: Sensitivity and replication (heterophilic mixing)

main_data = ['SC', 'SC', 'SC', 'FC', 'FC', 'FC']
supp_data = ['SC_s400', 'SC_L', 'SC_219', 'FC_s400', 'FC_L', 'FC_219']
names = ['HCP - 400', 'HCP - left', 'LAU - 219',
         'HCP - 400', 'HCP - left','LAU - 219']

correlations = {}
for main, supp, name in zip(main_data, supp_data, names):

    results_main = fn.load_data(
        f"results/heterophilic_assortativity/{main}_receptor.pickle")
    results_supp = fn.load_data(
        f"results/heterophilic_assortativity/{supp}_receptor.pickle")

    a_z_main = fn.fill_triu(results_main['a_z'])
    a_z_supp = fn.fill_triu(results_supp['a_z'])

    plt.figure(figsize=(2.3, 2.3))
    plt.scatter(x=a_z_supp,
                y=a_z_main,
                color='lightgray',
                s=8,
                rasterized=True)
    plt.xlabel(name)
    plt.ylabel("HCP - 800")
    X = a_z_main.flatten()
    Y = a_z_supp.flatten()

    pearsonr_results = pearsonr(X, Y)
    r, p = pearsonr_results
    CI = pearsonr_results.confidence_interval()
    df = len(X) - 2

    plt.title(f"r={r:.2f} | p={p:.5e}")

    plt.savefig(
        (f"figures/figure s8/{supp}.svg"),
        dpi=300)

    correlations[f'{supp}'] = (r, p, len(X)-2, CI)

# %% FIGURE S9: Homophilic ratios in the functional connectome

# Load results
FC_ratios = fn.load_data("results/local_mixing/FC_ratios.pickle")

# Find maximal + minimal ratios (for colormap) | set at 97.5 percentile
ratios_all = [FC_ratios[ann] for ann in human_annotations]
ratios_min = np.percentile(ratios_all, 2.5)
ratios_max = np.percentile(ratios_all, 97.5)

# plot homophilic ratios on brain surface
for ann in human_annotations:

    FC_ratio = FC_ratios[ann]

    lhannot, rhannot = fetch_schaefer2018()['800Parcels7Networks']
    surface_image, dot_image = fn.plot_homophilic_ratios(
        FC_ratio, ann, human_SC['coords'], lhannot=lhannot, rhannot=rhannot,
        noplot=human_SC['noplot'], order=human_SC['order'], vmin=ratios_min,
        vmax=ratios_max)

    surface_image.save_image(
        f"figures/figure s9/homophilic_ratio_surface_{ann}.png", mode='rgba')

    dot_image.savefig(f"figures/figure s9/homophilic_ratio_dot_{ann}.png",
                      dpi=600, transparent=True)

# %% FIGURE S10: Sensitivity and replication (homophilic ratios)

info = [[human_SC_s400, 'SC_s400', '', fetch_schaefer2018()['400Parcels7Networks'], 400],
        [human_SC_L, 'SC_L', 'L', fetch_schaefer2018()['800Parcels7Networks'], 800],
        [human_SC_219, 'SC_219', '', fetch_cammoun2012('fsaverage')['scale125'], 219],
        [human_SC_1000, 'SC_1000', '', fetch_cammoun2012('fsaverage')['scale500'], 1000]]

'''
For each connectome
'''

for connectome, label, hemi, (lhannot, rhannot), n_nodes in info:

    # Load results
    ratios = fn.load_data(f"results/local_mixing/{label}_ratios.pickle")

    # Find maximal + minimal ratios (for colormap) | set at 97.5 percentile
    ratios_all = [ratios[ann] for ann in human_annotations]
    ratios_min = np.percentile(ratios_all, 2.5)
    ratios_max = np.percentile(ratios_all, 97.5)

    # plot homophilic ratios on brain surface
    for ann in human_annotations:
        ratio = ratios[ann]

        surface_image, dot_image = fn.plot_homophilic_ratios(
            ratio, ann, connectome['coords'], lhannot, rhannot,
            connectome['noplot'], connectome['order'], ratios_min, ratios_max,
            hemi='L', n_nodes=n_nodes)

        surface_image.save_image(
            f"figures/figure s10/homophilic_ratio_surface_{label}_{ann}.png",
            mode='rgba')

        dot_image.savefig(
            f"figures/figure s10/homophilic_ratio_dot_{label}_{ann}.png",
            dpi=600, transparent=True)

# %% FIGURE S11: Homophilic ratios in the SC communities

'''
panel (a): communities of the structural connectome
'''

# Color for each community
colors = ['#b7df61ff', '#5ae858ff', '#58e8b3ff', '#fbbd45ff',
          '#61b9dfff', '#585de8ff', '#af58e8ff', '#df61bcff',
          '#e85858ff']

for i in range(9):
    node_size_aspect = 15
    scores = np.zeros((800))
    scores[human_SC['ci'] == i+1] = 1
    node_size = 25 * (10/5 + zscore(scores) + abs(zscore(scores).min()))
    fn.plot_network(human_SC['adj'],
                    human_SC['coords'][:, [0, 1]],
                    human_SC['adj'],
                    scores,
                    view_edge=False,
                    edge_alpha=0.10,
                    node_cmap=fn.get_cmap(['lightgray', colors[i]]),
                    s=node_size)
    plt.savefig(f"figures/figure s11/ci_{i}.png", dpi=300, transparent=True)

'''
panel (b): homophilic, geometric and topological properties of SC communities
'''

# Load homophilic ratios
mean_ratios = fn.load_data("results/local_mixing/SC_ratios.pickle")['mean']

# Compute node strength and mean connection distances
strength = human_SC['adj'].sum(axis=0)
mcd = np.average(human_SC['dist'], weights=human_SC['adj'], axis=0)

# Compute averaged scores in each community
ci = human_SC['ci']
n_ci = int(ci.max())
str_ci, ratios_ci, mcd_ci = np.zeros((n_ci)), np.zeros((n_ci)), np.zeros((n_ci))
for i in range(n_ci):
    str_ci[i] = np.mean(strength[ci == i+1])
    ratios_ci[i] = np.mean(mean_ratios[ci == i+1])
    mcd_ci[i] = np.mean(mcd[ci == i+1])

# Generate the colormap for the figure
colors = ['#b7df61ff', '#5ae858ff', '#58e8b3ff', '#fbbd45ff',
          '#61b9dfff', '#585de8ff', '#af58e8ff', '#df61bcff',
          '#e85858ff']

fig_str_MAD = plt.figure(figsize=(2.5, 2.5))
plt.scatter(str_ci,
            ratios_ci,
            c=np.arange(n_ci),
            s=75,
            cmap=fn.get_cmap(colors))
plt.xlabel("node strength")
plt.ylabel("conn/all ratio (average)")
plt.tight_layout()
plt.savefig("figures/figure s11/ratios_strength_communities.svg")

fig_mcl_MAD = plt.figure(figsize=(2.5, 2.5))
plt.scatter(mcd_ci,
            ratios_ci,
            c=np.arange(n_ci),
            s=75,
            cmap=fn.get_cmap(colors))
plt.xlabel("mean connection distance")
plt.ylabel("conn/all ratio (average)")
plt.tight_layout()
plt.savefig("figures/figure s11/ratios_mcd_communities.svg")

# %% FIGURE S12: Neurosynth correlations

neurosynth_results = fn.load_data("results/local_mixing/neurosynth.pickle")

r = neurosynth_results['r']
p = neurosynth_results['p']

r_order = np.argsort(r)
r_ordered = r[r_order]
terms_ordered = np.asarray(neurosynth_results['terms'])[r_order]

sigs = (p[r_order] < 0.05)
sigs_pos = (sigs) & (r_ordered > 0)
sigs_neg = (sigs) & (r_ordered < 0)

# negatives
r_negatives = r_ordered[r_ordered < 0]
colors = np.asarray(['white'] * len(r_negatives), dtype=object)
colors[sigs_neg[r_ordered < 0]] = '#f7fcf0ff'
fig_positive = plt.figure(figsize=(9.2, 3))
plt.bar(np.arange(len(r_negatives)),
        r_negatives,
        color=colors,
        edgecolor='black')
plt.xticks(np.arange(len(r_negatives)),
           terms_ordered[r_ordered < 0],
           rotation=90,
           fontsize=8)
plt.ylabel("Pearson's r")
plt.savefig("figures/figure s12/neurosynth_correlations_negative.svg")

# positives
r_positives = r_ordered[r_ordered > 0]
colors = np.asarray(['white'] * len(r_positives), dtype=object)
colors[sigs_pos[r_ordered > 0]] = '#084081ff'
fig_negative = plt.figure(figsize=(9.2, 3))
plt.bar(np.arange(len(r_positives)),
        r_positives,
        color=colors,
        edgecolor='black')
plt.xticks(np.arange(len(r_positives)),
           terms_ordered[r_ordered > 0],
           rotation=90,
           fontsize=8)
plt.ylabel("Pearson's r")
plt.savefig("figures/figure s12/neurosynth_correlations_positive.svg")
