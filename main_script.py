# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:52:55 2022

This is the main script of this repository. It can be used to reproduce the
results presented in "Assortative mixing in micro-architecturally annotated
brain connectomes".

The computation of some of the results presented in the paper takes a
long time. Pre-computed results are therefore saved in the results/ directory
of this repository.

The script is split into different cells (separated by `#%%`). Each cell
represent lines of codes used to load the data, compute a specific result, or
plot the elements of a specific figure:

DATA: Load data used in the experiments
RESULT 1: Standardized assortativity of micro-architectural annotations
RESULT 2: Assortative mixing of long-range connections
RESULTS 3: Heterophilic mixing
RESULTS 4: Homophilic ratios
FIGURE 1: Annotated connectomes
FIGURE 3: Standardized assortativity of micro-architectural annotations
FIGURE 4: Assortative mixing of long-range connections
FIGURE 5: Heterophilic mixing
FIGURE 6: Local assortative mixing

@author: Vincent Bazinet
"""

# IMPORT STATEMENTS
import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from netneurotools.plotting import plot_fsaverage
from tqdm import trange
from scipy.stats import zscore, pearsonr

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

#%% DATA: Load data used in the experiments

# Human connectomes
human_SC = fn.load_data("data/human_SC.pickle")
human_FC = fn.load_data("data/human_FC.pickle")
human_annotations = ['receptor_den', 'EI_ratio', 'genePC1', 't1t2', 'thi']

# Macaque connectome
scholtens = fn.load_data("data/macaque_scholtens.pickle")
macaque_annotations = ['t1t2', 'thi', 'neuron_den']

# Mouse connectome
oh = fn.load_data("data/mouse_oh.pickle")
mouse_annotations = ['genePC1']

# Additional datasets
receptor_densities = fn.load_data("data/receptor_densities.pickle")
laminar_thicknesses = fn.load_data("data/laminar_thicknesses.pickle")
neurosynth = fn.load_data("data/neurosynth.pickle")

#%% RESULT 1: Standardized assortativity of micro-architectural annotations

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

#%% RESULT 2: Assortative mixing of long-range connections

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

#%% RESULTS 3: Heterophilic mixing

result3_path = "results/heterophilic_assortativity"

# structural connectome (laminar thickness)
SC_laminar_results = fn.compute_heterophilic_assortativity(
    human_SC, laminar_thicknesses['data'].T,
    human_SC['spin_nulls'], laminar_thicknesses['names'])
fn.save_data(SC_laminar_results, f'{result3_path}/SC_laminar.pickle')

# structural connectome (receptor densities)
SC_receptor_results = fn.compute_heterophilic_assortativity(
    human_SC, receptor_densities['data'].T,
    human_SC['spin_nulls'], receptor_densities['names'])
fn.save_data(SC_receptor_results, f'{result3_path}/SC_receptor.pickle')

# functional connectome (laminar thickness)
FC_laminar_results = fn.compute_heterophilic_assortativity(
    human_FC, laminar_thicknesses['data'].T,
    human_FC['spin_nulls'], laminar_thicknesses['names'])
fn.save_data(FC_laminar_results, f'{result3_path}/FC_laminar.pickle')

# functional connectome (receptor densities)
FC_receptor_results = fn.compute_heterophilic_assortativity(
    human_FC, receptor_densities['data'].T,
    human_FC['spin_nulls'], receptor_densities['names'])
fn.save_data(FC_receptor_results, f'{result3_path}/FC_receptor.pickle')

#%% RESULTS 4: Homophilic ratios

# homophilic ratios
SC_ratios = fn.compute_homophilic_ratio(human_SC, human_annotations)
SC_ratios['mean'] = np.mean([SC_ratios[ann]
                             for ann in human_annotations], axis=0)
fn.save_data(SC_ratios, "results/local_mixing/SC_ratios.pickle")

# correlations with neurosynth
neurosynth_results = {}
neurosynth_results['r'] = np.zeros((123))
neurosynth_results['r_spin'] = np.zeros((123, 10000))

for i in trange(123):
    neurosynth_results['r'][i], _ = pearsonr(
        neurosynth['maps'][:,i], SC_ratios['mean'])
    for j in range(10000):
        spin = human_SC['spin_nulls'][:, j]
        neurosynth_results['r_spin'][i,j], _ = pearsonr(
            neurosynth['maps'][spin, i], SC_ratios['mean'])

neurosynth_results['p']  = np.zeros((123))
for i in range(123):
    neurosynth_results['p'][i] = fn.get_p_value(
        neurosynth_results['r_spin'][i, :], neurosynth_results['r'][i])
fn.save_data(neurosynth_results, "results/local_mixing/neurosynth.pickle")

#%% FIGURE 1: Annotated connectomes

'''
human (structural) connectome
'''

# threshold connectome (for visualization)
adj = human_SC['adj'].copy()
adj[adj < np.percentile(adj, 98)] = 0

# plot connectome
fn.plot_network(
    adj, human_SC['coords'][:, :2], adj, None, edge_alpha=0.5, s=15,
    figsize=(5,5), node_cmap="Greys", edge_cmap=cmaps['Reds_3'], linewidth=0.5)
save_path = "figures/figure 1/human (structural).png"
plt.savefig(save_path, dpi=600, rasterized=True, transparent=True)

'''
human (functional) connectome
'''

# threshold connectome (for visualization)
adj = human_FC['adj'].copy()
adj[adj < np.percentile(adj, 98.5)] = 0

# plot connectome
fn.plot_network(
    adj, human_FC['coords'][:, :2], adj, None, edge_alpha=0.5, s=15,
    figsize=(5,5), node_cmap="Greys", edge_cmap=cmaps['Blues_3'],
    linewidth=0.5)
save_path = "figures/figure 1/human (functional).png"
plt.savefig(save_path, dpi=600, rasterized=True, transparent=True)

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
    plt.savefig(save_path, dpi=600, rasterized=True, transparent=True)

'''
macaque (scholtens) connectome
'''

# bilaterize network (for visualization)
adj, coords = fn.bilaterize_network(
    scholtens['adj'], scholtens['coords'], symmetry_axis=0)

# plot connectome
fn.plot_network(
    adj, coords[:, :2], adj, None, linewidth=1.5, edge_alpha=0.1,
    s=100, figsize=(5,5), node_cmap="Greys",edge_cmap="Greens")
plt.savefig("figures/figure 1/macaque (scholtens).png", dpi=600,
            rasterized=True, transparent=True)

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
                transparent=True, dpi=600, rasterized=True)

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
    adj, coords[:, [0,2]], None, None, linewidth=1.5, edge_alpha=0.01, s=50,
    figsize=(5,5), node_cmap="Greys", edge_vmin=0, edges_color="#3f007d64")
plt.savefig("figures/figure 1/mouse (oh).png",
            dpi=600, rasterized=True, transparent=True)

'''
mouse (oh) annotations
'''

for ann in mouse_annotations:

    # bilaterize annotation scores (for visualization)
    scores = np.concatenate((oh[ann], oh[ann]))

    fn.plot_network(
        None, coords[:, [0,2]], None, scores, linewidth=1.5, edge_alpha=0.01,
        s=50, figsize=(5,5), view_edge=False, node_cmap=cmaps['Spectral_11_r'])
    plt.savefig(f"figures/figure 1/mouse - {ann}.png",
                transparent=True, dpi=600, rasterized=True)

#%% FIGURE 3: Standardized assortativity of micro-architectural annotations

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
panel (b): barplots with stanardized assortativity results
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
                               sig_colors,null_types)
fig.savefig("figures/figure 3/barplot.svg")

#%% FIGURE 4: Assortative mixing of long-range connections

percent_kept = np.arange(5, 100, 5)

# human (structural)
fig = fn.plot_assortativity_thresholded(
    "human_SC", 'spin', human_annotations, percent_kept,
    ['#28c828ff', '#1a1ad6ff', '#f2b701ff', '#c828c6ff', '#1ad6d4ff'],
    ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff'])
fig.savefig("figures/figure 4/lineplot_human_SC.svg")

# human (functional)
fig = fn.plot_assortativity_thresholded(
    "human_FC", 'spin', human_annotations, percent_kept,
    ['#28c828ff', '#1a1ad6ff', '#f2b701ff', '#c828c6ff', '#1ad6d4ff'],
    ['#cdf5cdff', '#cacaf8ff', '#feeebaff', '#f5cdf5ff', '#caf8f8ff'])
fig.savefig("figures/figure 4/lineplot_human_FC.svg")

# macaque (scholtens)
fig = fn.plot_assortativity_thresholded(
    "scholtens", 'moran', macaque_annotations, percent_kept,
    ['#c828c6ff', '#1ad6d4ff', '#d61a1aff'],
    ['#f5cdf5ff', '#caf8f8ff', '#f8cacaff'])
fig.savefig("figures/figure 4/lineplot_macaque.svg")

# mouse (oh)
fig = fn.plot_assortativity_thresholded(
    "oh", 'moran', mouse_annotations, percent_kept,
    ['#f2b701ff'], ['#feeebaff'])
fig.savefig("figures/figure 4/lineplot_mouse.svg")

#%% FIGURE 5: Heterophilic mixing

path_results_fig5 = "results/heterophilic_assortativity"

'''
Identify min and max assortativity values across network + annotation types
'''

test = fn.generate_moran_nulls(human_FC['t1t2'], human_FC['dist'], 1000)

m = 0
for matrix_name in ['SC_laminar', 'FC_laminar', 'SC_receptor', 'FC_receptor']:
    matrix = fn.load_data(f'{path_results_fig5}/{matrix_name}.pickle')
    max_assort = np.abs(matrix['assort_z']).max()
    if max_assort > m:
        m = max_assort

'''
panel (b): mixing matrices (laminar thickness)
'''

layer_names = laminar_thicknesses['names']

# structural connectome
SC_laminar = fn.load_data(f'{path_results_fig5}/SC_laminar.pickle')
fn.plot_heatmap(SC_laminar['assort_z'], layer_names, layer_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=2, figsize=(3.4, 3.4),
                sigs=SC_laminar['assort_p'] < 0.05, text_size=17)
plt.savefig("figures/figure 5/heterophilic_matrix_SC_laminar.svg")

# functional connectome
FC_laminar = fn.load_data(f'{path_results_fig5}/FC_laminar.pickle')
fn.plot_heatmap(FC_laminar['assort_z'], layer_names, layer_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=2, figsize=(3.4, 3.4),
                sigs=FC_laminar['assort_p'] < 0.05, text_size=17)
plt.savefig("figures/figure 5/heterophilic_matrix_FC_laminar.svg")

'''
panel (b): mixing matrices (receptor densities)
'''

receptor_names = receptor_densities['names']

# structural connectome
SC_receptor = fn.load_data(f'{path_results_fig5}/SC_receptor.pickle')
fn.plot_heatmap(SC_receptor['assort_z'], receptor_names, receptor_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=1, figsize=(5, 5),
                sigs=SC_receptor['assort_p'] < 0.05)
plt.savefig("figures/figure 5/heterophilic_matrix_SC_receptors.svg")

# functional connectome
FC_receptor = fn.load_data(f'{path_results_fig5}/FC_receptor.pickle')
fn.plot_heatmap(FC_receptor['assort_z'], receptor_names, receptor_names,
                cbarlabel='z-assortativity', cmap=cmaps['RdBu_11_r'],
                vmin=-m, vmax=m, grid_width=1, figsize=(5, 5),
                sigs=FC_receptor['assort_p'] < 0.05)
plt.savefig("figures/figure 5/heterophilic_matrix_FC_receptors.svg")

'''
panel (c): scatterplot of assortativity results for SC and FC
'''

fn.plot_SC_FC_heterophilic_comparison(
    SC_receptor, FC_receptor, SC_laminar, FC_laminar)
plt.savefig("figures/figure 5/scatterplot_SC_FC.svg")

#%% FIGURE 6: Local assortative mixing

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
    im = plot_fsaverage(SC_ratio,
                        lhannot=human_SC['lhannot'],
                        rhannot=human_SC['rhannot'],
                        noplot=human_SC['noplot'],
                        order=human_SC['order'],
                        views=['lateral', 'm'],
                        vmin=ratios_min,
                        vmax=ratios_max,
                        colormap=cmaps['GnBu_9'],
                        data_kws={'representation': 'wireframe',
                                  'line_width': 4.0}
                        )
    im.save_image(f"figures/figure 6/homophilic_ratio_surface_{ann}.png",
                  mode='rgba')

# plot homophilic ratios on dotted brain
for ann in human_annotations:
    scores = SC_ratios[ann]
    size_change = abs(zscore(scores))
    size_change[size_change > 5] = 5
    size = 40 + (10 * size_change)
    fn.plot_network(None, human_SC['coords'][:, :2], None, scores,
                    s=size, view_edge=False, node_cmap=cmaps['GnBu_9'],
                    node_vmin=ratios_min, node_vmax=ratios_max)
    plt.savefig(f"figures/figure 6/homophilic_ratio_dot_{ann}.png", dpi=600,
                transparent=True)

'''
panel (d): topology and geometry
'''

ratios_mean = np.mean([SC_ratios[ann] for ann in human_annotations], axis=0)
strength = human_SC['adj'].sum(axis=0)
mcd = np.average(human_SC['dist'], weights=human_SC['adj'], axis=0)

# Node strength
fig = fn.plot_regression(
    strength, ratios_mean, x_label="node strength",
    y_label='homophilic ratio (average)', permutations=human_SC['spin_nulls'])
fig.savefig("figures/figure 6/scatterplot_str.svg", dpi=300)

# Mean connection length
fig = fn.plot_regression(
    mcd, ratios_mean, x_label="mean connection distance",
    y_label='homophilic ratio (average)', permutations=human_SC['spin_nulls'])
plt.savefig("figures/figure 6/scatterplot_mcd.svg", dpi=300)

'''
panel (e): neurosynth
'''

neurosynth_results = fn.load_data("results/local_mixing/neurosynth.pickle")

plt.figure()
sigs = (neurosynth_results['p'] < 0.05)
r_sigs = neurosynth_results['r'][sigs]
terms_sigs = np.array(neurosynth['terms'])[sigs]
r_order = np.argsort(r_sigs)
plt.figure()
for i in range(len(r_sigs)):
    if r_sigs[r_order][i] < 0:
        color = '#f7fcf0ff'
    else:
        color = '#084081ff'
    plt.bar(i, r_sigs[r_order][i], color=color, edgecolor='black')
plt.xticks(np.arange(len(r_sigs)),
           terms_sigs[r_order],
           rotation=90)
plt.tight_layout()
plt.ylabel("Pearson's r")
plt.savefig("figures/figure 6/barplot_neurosynth_r.svg")