# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:53:22 2022

This file contains function used to load the required data, run the analyses
and plot the figures presented in 'Assortative mixing in micro-architecturally
annotated brain connectomes'.

@author: Vincent Bazinet
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from tqdm import trange, tqdm
from brainspace.null_models.moran import MoranRandomization
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from palettable.colorbrewer.diverging import Spectral_11_r, RdBu_11_r
from palettable.colorbrewer.sequential import Reds_3, Blues_3, GnBu_9
from scipy.stats import pearsonr

'''
ASSORTATIVITY FUNCTIONS
'''


def weighted_assort(A, M, N=None, directed=True, normalize=True):
    '''
    Function to compute the weighted Pearson correlation between the attributes
    of the nodes connected by edges in a network (i.e. weighted assortativity).
    This function also works for binary networks.

    Parameters
    ----------
    A : (n,n) ndarray
        Adjacency matrix of our network.
    M : (n,) ndarray
        Vector of nodal attributes.
    N : (n,) ndarray
        Second vector of nodal attributes (optional)
    directed: bool
        Whether the network is directed or not. When the network is not
        directed, setting this parameter to False will increase the speed of
        the computations.
    normalize: bool
        If False, the adjacency weights won't be normalized to make its weights
        sum to 1. This should only be set to False if the matrix has been
        normalized already. Otherwise, the result will not be the assortativity
        coefficent. This is useful when we want to compute the assortativity
        of thousands annotations in a row. In that case, not having to
        normalize the adjacency matrix each time makes the function much
        faster.

    Returns
    -------
    ga : float
        Weighted assortativity of our network, with respect to the vector
        of attributes
    '''

    if (directed) and (N is None):
        N = M

    # Normalize the adjacency matrix to make weights sum to 1
    if normalize:
        A = A / A.sum(axis=None)

    # zscores of in-annotations
    k_in = A.sum(axis=0)
    mean_in = np.sum(k_in * M)
    sd_in = np.sqrt(np.sum(k_in * ((M-mean_in)**2)))
    z_in = (M - mean_in) / sd_in

    # zscores of out-annotations (if directed or N is not None)
    if N is not None:
        k_out = A.sum(axis=1)
        mean_out = np.sum(k_out * N)
        sd_out = np.sqrt(np.sum(k_out * ((N-mean_out)**2)))
        z_out = (N - mean_out) / sd_out
    else:
        z_out = z_in

    # Compute the weighted assortativity as a sum of z-scores
    ga = (z_in[np.newaxis, :] * z_out[:, np.newaxis] * A).sum()

    return ga


def wei_assort_batch(A, M_all, N_all=None, n_batch=100, directed=True):
    '''
    Function to compute the weighted assortativity of a "batch" of attributes
    on a single network.

    Parameters
    ----------
    A : (n, n) ndarray
        Adjacency matrix
    M_all : (m, n) ndarray
        Attributes
    n_batch: int
        Number of attribute in each batch.
    directed: bool
        Whether the network is directed or not. When the network is not
        directed, setting this parameter to False will increase the speed of
        the computations.

    Returns
    -------
    '''
    n_attributes, n_nodes = M_all.shape
    ga = np.array([])

    # Create batches of annotations
    if N_all is not None:
        M_batches = zip(np.array_split(M_all, n_batch),
                        np.array_split(N_all, n_batch))
    elif directed:
        N_all = True
        M_batches = zip(np.array_split(M_all, n_batch),
                        np.array_split(M_all, n_batch))
    else:
        M_batches = np.array_split(M_all, n_batch)

    # Normalize the adjacency matrix to make weights sum to 1
    A = A / A.sum(axis=None)

    # Compute in- and out- degree (if directed)
    k_in = A.sum(axis=0)
    if (directed) or (N_all is not None):
        k_out = A.sum(axis=1)

    for M in M_batches:
        if N_all is not None:
            M, N = M
        # Z-score of in-annotations
        n_att, _ = M.shape
        mean_in = (k_in[np.newaxis, :] * M).sum(axis=1)
        var_in = (k_in[np.newaxis, :] * ((M - mean_in[:, np.newaxis])**2)).sum(axis=1)  # noqa
        sd_in = np.sqrt(var_in)
        z_in = (M - mean_in[:, np.newaxis]) / sd_in[:, np.newaxis]
        # Z-score of out-annotations
        if N_all is not None:
            n_att, _ = N.shape
            mean_out = (k_out[np.newaxis, :] * N).sum(axis=1)
            var_out = (k_out[np.newaxis, :] * ((N - mean_out[:, np.newaxis])**2)).sum(axis=1)  # noqa
            sd_out = np.sqrt(var_out)
            z_out = (N - mean_out[:, np.newaxis]) / sd_out[:, np.newaxis]
        else:
            z_out = z_in
        # Compute assortativity
        ga_batch = A[np.newaxis, :, :] * z_out[:, :, np.newaxis] * z_in[:, np.newaxis, :]  # noqa
        ga_batch = ga_batch.sum(axis=(1, 2))
        # Add assortativity results to ga
        ga = np.concatenate((ga, ga_batch), axis=0)

    return ga


'''
UTILITY FUNCTIONS
'''


def load_data(path):
    '''
    Utility function to load pickled dictionary containing the data used in
    these experiments.

    Parameters
    ----------
    path: str
        File path to the pickle file to be loaded.

    Returns
    -------
    data: dict
        Dictionary containing the data used in these experiments
    '''

    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data


def save_data(data, path):
    '''
    Utility function to load pickled dictionary containing the data used in
    these experiments.

    Parameters
    ----------
    data: dict
        Dictionary storing the data that we want to save.
    path: str
        File path to the pickle file to be loaded
    '''

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def standardize_scores(surr, emp, axis=None, ignore_nan=False):
    '''
    Utility function to standardize a score relative to a null distribution.

    Parameters
    ----------
    perm: array-like
        Null distribution of scores.
    emp: float
        Empirical score.
    '''
    if ignore_nan:
        return (emp - np.nanmean(surr, axis=axis)) / np.nanstd(surr, axis=axis)
    else:
        return (emp - surr.mean(axis=axis)) / surr.std(axis=axis)


def get_p_value(perm, emp):
    '''
    Utility function to compute the p-value of a score, relative to a null
    distribution

    Parameters
    ----------
    perm: array-like
        Null distribution of (permuted) scores.
    emp: float
        Empirical score.
    '''
    k = len(perm)
    return len(np.where(abs(perm-np.mean(perm)) > abs(emp-np.mean(perm)))[0])/k


def get_corr_spin_p(X, Y, spins):
    '''
    Function to compute the p-value of a correlation score compared to spun
    distributions

    Parameters
    ----------
    X: (n,) ndarray
        Independent variable.
    Y: (n,) ndarray
        Dependent variable
    spins: (n, n_spin) ndarray
        Permutations used to compute the p-value of the correlation.

    Returns
    -------
    p_spin: float
        p-value of the correlation.
    '''

    N_nodes, N_spins = spins.shape
    emp_corr, _ = pearsonr(X, Y)
    spin_corr = np.zeros((N_spins))
    for i in range(N_spins):
        spin_corr[i], _ = pearsonr(X[spins[:, i]], Y)
    p_spin = get_p_value(spin_corr, emp_corr)

    return p_spin


'''
VISUALIZATION FUNCTIONS
'''


def get_colormaps():
    '''
    Utility function that loads colormaps from the palettable module into a
    dictionary

    Returns
    -------
    cmaps: dict
        Dictionary containing matplotlib colormaps
    '''

    cmaps = {}

    # Colorbrewer | Diverging
    cmaps['Spectral_11_r'] = Spectral_11_r.mpl_colormap
    cmaps['RdBu_11_r'] = RdBu_11_r.mpl_colormap

    # Colorbrewer | Sequential
    cmaps['Reds_3'] = Reds_3.mpl_colormap
    cmaps['Blues_3'] = Blues_3.mpl_colormap
    cmaps['GnBu_9'] = GnBu_9.mpl_colormap

    return cmaps


def plot_network(A, coords, edge_scores, node_scores, edge_cmap="Greys",
                 node_cmap="viridis", edge_alpha=0.25, node_alpha=1,
                 edge_vmin=None, edge_vmax=None, node_vmin=None,
                 node_vmax=None, nodes_color='black', edges_color='black',
                 linewidth=0.25, s=100, view_edge=True, figsize=None):
    '''
    Function to draw (plot) a network of nodes and edges.

    Parameters
    ----------
    A : (n, n) ndarray
        Array storing the adjacency matrix of the network. 'n' is the
        number of nodes in the network.
    coords : (n, 3) ndarray
        Coordinates of the network's nodes.
    edge_scores: (n,n) ndarray
        Array storing edge scores for individual edges in the network. These
        scores are used to color the edges.
    node_scores : (n) ndarray
        Array storing node scores for individual nodes in the network. These
        scores are used to color the nodes.
    edge_cmap, node_cmap: str
        Colormaps from matplotlib.
    edge_alpha, node_alpha: float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque)
    edge_vmin, edge_vmax, node_vmin, node_vmax: float, optional
        Minimal and maximal values of the node and edge colors. If None,
        the min and max of edge_scores and node_scores respectively are used.
        Default: `None`
    nodes_color, edges_color: str
        Color to be used to plot the network's nodes and edges if edge_scores
        or node_scores are none.
    linewidth: float
        Width of the edges.
    s: float or array-like
        Size the nodes.
    view_edge: bool
        If true, network edges are shown.
    figsize: (float, float)
        Width and height of the figure, in inches.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    ax: matplotlib.axes.Axes instance
        Ax instance of the drawn network.
    '''

    if figsize is None:
        figsize = (10, 10)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the edges
    if view_edge:

        # Identify edges in the network
        edges = np.where(A > 0)

        # Get the color of the edges
        if edge_scores is None:
            edge_colors = np.full((len(edges[0])), edges_color, dtype="<U10")
        else:
            edge_colors = cm.get_cmap(edge_cmap)(
                mpl.colors.Normalize(edge_vmin, edge_vmax)(edge_scores[edges]))

        # Plot the edges
        for edge_i, edge_j, c in zip(edges[0], edges[1], edge_colors):

            x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
            y1, y2 = coords[edge_i, 1], coords[edge_j, 1]

            ax.plot([x1, x2], [y1, y2], c=c, linewidth=linewidth,
                    alpha=edge_alpha, zorder=0)

    # Get the color of the nodes
    if node_scores is None:
        node_scores = nodes_color
    node_colors = node_scores

    # plot the nodes
    ax.scatter(
        coords[:, 0], coords[:, 1], c=node_colors,
        edgecolors='none', cmap=node_cmap, vmin=node_vmin,
        vmax=node_vmax, alpha=node_alpha, s=s, zorder=1)
    ax.set_aspect('equal')

    ax.axis('off')

    return fig, ax


def bilaterize_network(A, coords, symmetry_axis=0, between_hemi_dist=0):
    '''
    Function to bilaterize a single-hemisphere connectome (i.e. duplicate the
    number of nodes and the connectivity of the network)

    Parameters
    ----------
    A: (n, n) ndarray
        Adjacency matrix of the single-hemisphere connectome, where `n` is the
        number of nodes in this single-hemispheric connectome
    coords: (n, 3) ndarray
        Coordinates of the nodes in the single-hemisphere connectome
    symmetry_axis: int
        Axis of symmetry along which the network is bilaterized
    between_hemi_dist = float
        Distance between the coordinates of the two hemisphere

    Returns
    -------
    A_bil: (2*n, 2*n) ndarray
        Bilaterized adjacency matrix
    coords_bil: (2*n, 3) ndarray
        Coordinate of the nodes in the bilaterized connectome. Coordinates are
        mirrored along the symmetry axis.

    '''

    n_nodes = len(A)

    A_bil = np.zeros((n_nodes * 2, n_nodes * 2))
    A_bil[:n_nodes, :n_nodes] = A.copy()
    A_bil[n_nodes:, n_nodes:] = A.copy()

    coords_bil = np.zeros((n_nodes * 2, 3))
    coords_bil[:n_nodes] = coords.copy()
    coords_bil[n_nodes:] = coords.copy()
    coords_bil[:n_nodes, 2] += between_hemi_dist
    coords_bil[n_nodes:, symmetry_axis] = -coords_bil[:n_nodes, symmetry_axis]

    return A_bil, coords_bil


def assortativity_boxplot(network_name, null_type, annotations, figsize=(3, 2),
                          face_color='white', edge_color='black'):
    '''
    Function to plot the boxplots showing the distribution of null
    assortativity results, relative to each assortativity result obtained with
    empirical annotations. This function relies on the results stored
    in the `results/standardized_assortativity` folder.

    Parameters
    ----------
    network_name: str
        Name of the network. This is used to load the necessary results in
        stored in `results/standardized_assortativity/{network_name}.pickle`
    null_type: str
        Type of the null model used to compute the null distribution of
        assortativity results.
    annotations: list
        List of annotation names that we want to include in the figure.
    figsize: (float, float)
        Width and height of the figure, in inches.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    path2results = f"results/standardized_assortativity/{network_name}.pickle"
    results = load_data(path2results)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i, ann in enumerate(annotations):

        ax.scatter(i+1,
                   results[ann]['assort'],
                   c=edge_color,
                   s=50)

        bplot = ax.boxplot(results[ann][f'assort_{null_type}'],
                           positions=[i+1],
                           widths=0.5,
                           patch_artist=True,
                           medianprops=dict(color='black'),
                           flierprops=dict(marker='+',
                                           markerfacecolor='lightgray',
                                           markeredgecolor='lightgray'),
                           showcaps=False,
                           zorder=0)

        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(bplot[element], color=edge_color)

        for patch in bplot['boxes']:
            patch.set_facecolor(face_color)

    sns.despine()
    ax.set_ylabel("assortativity")
    ax.set_xlabel("annotation")
    ax.set_xticklabels(annotations)

    return fig


def assortativity_barplot(results, annotation_labels, non_sig_colors,
                          sig_colors, null_types):
    '''
    Function to plot the barplot showing the standardized assortativity
    results of all the annotations, across all networks.This function relies
    on the results stored in the `results/standardized_assortativity` folder.

    Parameters
    ----------
    results: list
        List of assortativity results for each annotation.
    annotation_labels: list
        List of annotation labels for each annotation.
    non_sig_colors: list of colors
        List of colors used to color each barplot that are non-significant,
        according to the network associated with it.
    sig_color: list of colors
        List of colors used to color each barplot that are significant,
        according to the network associated with it.
    null_type: str
        Type of the null model used to compute the null distribution of
        assortativity results.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    n_annotations = len(results)
    figsize = (n_annotations/2, 3)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i, (ann, null_type) in enumerate(zip(results, null_types)):

        if ann['assort_p'] < 0.05:
            color = sig_colors[i]
        else:
            color = non_sig_colors[i]

        plt.bar(i, ann['assort_z'], width=0.5, color=color,
                edgecolor=color, zorder=1)
    plt.plot([0, n_annotations], [0, 0], color='lightgray', linestyle='dashed',
             zorder=0)
    ax.set_ylabel("z-assortativity")
    ax.set_xlabel("annotation")
    ax.set_xticks(np.arange(len(results)), labels=annotation_labels,
                  rotation='vertical')
    plt.tight_layout()
    sns.despine()

    return fig


def plot_assortativity_thresholded(network_name, null_type, annotations,
                                   percent_kept, sig_colors, non_sig_colors):
    '''
    Function to plot lineplots of the standardized assortativity of annotations
    as a function of the percentile of short-range connections removed from the
    network.

    Parameters:
    ----------
    network_name: str
        Name of the network. This is used to load the necessary results in
        stored in `results/assortativity_thresholded/{network_name}.pickle`.
    null_type: str
        Type of the null model used to compute the null distribution of
        assortativity results.
    annotations: list
        List of annotation names that we want to include in the figure.
    percent_kept: array-like
        List of percentile values indicating the percentile of connections
        that we want to keep when thresholding the network.
    sig_colors: list
        List of colors used to indicate significance, for each annotation
    non_sig_colors: list
        List of colors used to indicate non-significance, for each annotation.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    percent_removed = 100 - percent_kept
    n_box = len(percent_kept)

    results = load_data(
        f"results/assortativity_thresholded/{network_name}.pickle")

    fig = plt.figure(figsize=(3.9, 1.8))
    for i, key in enumerate(annotations):

        assort_p = results[key]['assort_all_p']
        assort_z = results[key]['assort_all_z']

        # Set color
        color = np.zeros((n_box), dtype='object')
        color[:] = non_sig_colors[i]
        color[assort_p < 0.05] = sig_colors[i]

        # Plot trajectory lines
        plt.plot(percent_removed,
                 assort_z,
                 color=non_sig_colors[i],
                 zorder=0)

        # Plot scatterplot markers
        plt.scatter(percent_removed,
                    assort_z,
                    color=color,
                    s=10,
                    zorder=1,
                    label=key)

        # Plot dashed line at 0
        plt.plot(percent_removed,
                 np.zeros((n_box)),
                 linestyle='dashed',
                 color='lightgray')

    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    sns.despine()
    plt.legend()

    return fig


def plot_regression(X, Y, x_label=None, y_label=None, permutations=None):
    '''
    Function to plot a scatterplot showing the relationship between a variable
    X and a variable Y as well as the regression line of this relationship.

    Paramaters:
    ----------
    X: (n,) ndarray
        Independent variable.
    Y: (n,) ndarray
        Dependent variable
    x_label: str
        Label of the x-axis
    y_label: str
        Label of the y-axis
    permutations: (n, n_perm) ndarray
        Permutations used to compute the significance of the relatiosnhip.

    Returns:
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    r, _ = pearsonr(X, Y)
    if permutations is not None:
        p_spin = get_corr_spin_p(X, Y, permutations)
    fig = plt.figure(figsize=(3, 3))
    sns.regplot(
        X, Y, color='black', truncate=False,
        scatter_kws={'s': 5, 'rasterized': True,
                     'alpha': 0.5, 'edgecolor': 'none'}
                )
    plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if permutations is not None:
        plt.title(f"r={r:.2f}; p_spin={p_spin:.4f}")
    else:
        plt.title(f"r={r:.2f}")
    plt.tight_layout()

    return fig


def plot_heatmap(values, xlabels, ylabels, cbarlabel="values",
                 cmap="viridis", vmin=None, vmax=None, grid_width=3,
                 figsize=None, text_size=12, sigs=None):
    '''
    Function to plot a heatmap

    Parameters
    ----------
    values: ndarray
        Array storing the values displayed in the heatmaps
    xlabels, ylabels: list
        List of labels for each x-tick or y-tick
    cbarlabel: str
        Label of the colorbar
    cmap: str
        Colormap
    vmin, vmax: float
        Minimum and maximum values for colorbar.
    grid_width: float
        Width of the grid lines between each entry of the heatmap.
    figsize: (float, float)
        Width and height of the figure, in inches.
    text_size: float
        Size of the asterisks used to denote significance.
    sigs: ndarray of bool
        Matrix of boolean values indicating which scores in the `values` matrix
        are significant. Significant scores will be denoted with an asterisk.
    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    fig, ax = plt.subplots(dpi=100, figsize=figsize)

    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    # Plot the heatmap
    im = ax.imshow(values, aspect='equal', vmin=vmin, vmax=vmax, cmap=cmap)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

    # show each ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(values.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=grid_width)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add stars if significants
    if sigs is not None:
        for i in range(sigs.shape[0]):
            for j in range(sigs.shape[1]):
                if sigs[i, j]:
                    im.axes.text(j, i, '*', horizontalalignment='center',
                                 verticalalignment='center', color="white",
                                 fontsize=text_size)

    fig.tight_layout()

    return fig


def plot_SC_FC_heterophilic_comparison(SC_receptors, FC_receptors, SC_layers,
                                       FC_layers):
    '''
    Function to plot the scatterplot comparing heterophilic mixing in the
    structural and functional connectomes. Data points that are significant
    on both data axes are colored in red if they are positive, and in blue
    if they are negative.

    Parameters
    ----------
    SC_receptors: dict
        Dictionary storing the heterophilic mixing results for the receptor
        data, for the structural connectome.
    FC_receptors: dict
        Dictionary storing the heterophilic mixing results for the receptor
        data, for the functional connectome.
    SC_layers: dict
        Dictionary storing the heterophilic mixing results for the laminar
        data, for the structural connectome.
    FC_layers: dict
        Dictionary storing the heterophilic mixing results for the laminar
        data, for the functional connectome.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    def scatterplot_significance_colored(X, Y):

        plt.scatter(
            x=X['assort_z'], y=Y['assort_z'],
            color='lightgray', s=8, rasterized=True)
        pos_sig_SC = (X['assort_p'] < 0.05) & (X['assort_z'] > 0)
        pos_sig_FC = (Y['assort_p'] < 0.05) & (Y['assort_z'] > 0)
        plt.scatter(x=X['assort_z'][(pos_sig_SC) & (pos_sig_FC)],
                    y=Y['assort_z'][(pos_sig_SC) & (pos_sig_FC)],
                    color='#67001F',  s=8, rasterized=True)
        neg_sig_SC = (X['assort_p'] < 0.05) & (X['assort_z'] < 0)
        neg_sig_FC = (Y['assort_p'] < 0.05) & (Y['assort_z'] < 0)
        plt.scatter(x=X['assort_z'][(neg_sig_SC) & (neg_sig_FC)],
                    y=Y['assort_z'][(neg_sig_SC) & (neg_sig_FC)],
                    color='#053061', s=8, rasterized=True)

    fig = plt.figure(figsize=(2.3, 2.3))

    # Plot receptors results
    scatterplot_significance_colored(SC_receptors, FC_receptors)

    # Plot layer results
    scatterplot_significance_colored(SC_layers, FC_layers)

    plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    plt.xlabel("z-assortativity (SC)")
    plt.ylabel("z-assortativity (FC)")

    # Compute correlation
    SC_a_z_all = np.concatenate(
        (SC_receptors['assort_z'].flatten(),
         SC_layers['assort_z'].flatten()),
        axis=0)
    FC_a_z_all = np.concatenate(
        (FC_receptors['assort_z'].flatten(),
         FC_layers['assort_z'].flatten()),
        axis=0)
    r, _ = pearsonr(SC_a_z_all, FC_a_z_all)
    plt.title(f"r={r:.2f}")

    return fig


'''
RESULTS FUNCTIONS
'''


def generate_moran_nulls(scores, dist, n_nulls):
    '''
    Function to generate Moran nulls. This function relies on the brainspace
    toolbox.

    Parameters
    ----------
    scores: (n,) ndarray
        Vector of annotation scores for individual brain regions.
    dist: (n, n) ndarray
        Matrix of distances between each pair of brain regions.
    n_nulls: int
        Number of null annotations to generate.

    Returns
    -------
    nulls: (n_nulls, n) ndarray
        Array of null annotations

    '''
    w = spatial_weights(dist)

    rand_seed = np.random.default_rng().integers(0, 2**32)

    moranRandom = MoranRandomization(tol=1e-5, n_rep=n_nulls,
                                     random_state=rand_seed)
    moranRandom.fit(w)
    nulls = moranRandom.randomize(scores)

    return nulls


def spatial_weights(dist):
    '''
    Function to get the spatial weight matrix of our network used to generate
    moran spatial nulls.

    Parameters
    ----------
    dist: (n, n) ndarray
        Matrix storing the euclidean distances between each node in the
        network.

    Returns
    -------
    w: (n, n) ndarray
        Matrix storing the spatial weights for each pair of node in the
        network.
    '''

    w = dist.copy()
    w[w > 0] = 1/w[w > 0]
    np.fill_diagonal(w, 1)

    return w


def compute_standardized_assortativity(network, null_type, annotations,
                                       directed=True):
    '''
    Function to compute the standardized assortativity of a list of annotation
    for a given network.

    Parameters
    ----------
    network: dict
        Dictionary storing relevant information about the network
    null_type: str
        Type of null used to compute the standardized assortativity
    annotations: list
        List of annotation names. These names should correspond to keys in the
        network dictionary.
    directed: bool
        Whether the network is directed or not. When the network is not
        directed, setting this parameter to False will increase the speed of
        the computations.

    Returns
    -------
    results: dict
        Dictionary storing the results computed with this function. They
        include, for each annotation:
        `assort`: assortativity score
        `assort{null_type}`: assortativity scores computed using the surrogate
            annotations.
        `assort_z`: standardized assortativity scores relative to the
            assortativity scores computed using the surrogate annotations.
        `assort_p`: p-value of the assortativity score.

    '''
    results = {}

    for ann in annotations:

        print(f"computing standardized assortativity for: `{ann}`")

        results[ann] = {}

        # assortativity with empirical annotations
        results[ann]['assort'] = weighted_assort(
            network['adj'], network[ann], directed=directed)

        # assortativity with surrogate annotations
        if null_type == 'spin':
            nulls = network[ann][network['spin_nulls']].T
        elif null_type == 'moran':
            nulls = generate_moran_nulls(network[ann], network['dist'], 10000)
        results[ann][f'assort_{null_type}'] = wei_assort_batch(
            network['adj'], nulls, n_batch=100, directed=directed)

        # compute standardized assortativity + p-values
        results[ann]['assort_z'] = standardize_scores(
            results[ann][f'assort_{null_type}'], results[ann]['assort'])
        results[ann]['assort_p'] = get_p_value(
            results[ann][f'assort_{null_type}'], results[ann]['assort'])

    return results


def compute_assortativity_thresholded(network, null_type, annotations,
                                      percent_kept, directed=True):
    '''
    Function to compute the standardized assortativity of a list of annotation
    for a given network that is thresholded on the basis of the length
    of its connections (i.e. short-range connections are removed).

    Parameters
    ----------
    network: dict
        Dictionary storing relevant information about the network
    null_type: str
        Type of null used to compute the standardized assortativity
    annotations: list
        List of annotation names. These names should correspond to keys in the
        network dictionary.
    percent_kept: array-like
        List of percentile values indicating the percentile of connections
        that we want to keep when thresholding the network.
    directed: bool
        Whether the network is directed or not. When the network is not
        directed, setting this parameter to False will increase the speed of
        the computations.

    Returns
    -------
    results: dict
        Dictionary storing the results computed with this function. They
        include, for each annotation:
        `assort_all`: Assortativity scores, for each thresholded connectome.
        `assort_all_{null_type}`: Assortativity scores of the surrogate
            annotations, for each thresholded connectome.
        `assort_all_z`: Standardized assortativity scores, for each thresholded
            connectome.
        `assort_all_p`: p-values the assortativity scores, for each thresholded
            connectome.

    '''
    n_bins = len(percent_kept)

    results = {}
    for ann in annotations:
        results[ann] = {}
        results[ann]['assort_all'] = np.zeros((n_bins))
        results[ann][f'assort_all_{null_type}'] = np.zeros((n_bins, 10000))
        results[ann]['assort_all_p'] = np.zeros((n_bins))
        results[ann]['assort_all_z'] = np.zeros((n_bins))

        scores = network[ann]
        for i, percent in enumerate(tqdm(percent_kept)):

            # threshold connectomes (remove short-range connections)
            A = network['adj'].copy()
            threshold = np.percentile(
                network['dist'][network['adj'] > 0], 100 - percent)
            A[network['dist'] < threshold] = 0

            # compute assortativity with empirical annotation
            results[ann]['assort_all'][i] = weighted_assort(A, scores)

            # compute assortativity with surrogate annotations
            if null_type == 'spin':
                nulls = network[ann][network['spin_nulls']].T
            elif null_type == 'moran':
                nulls = generate_moran_nulls(
                    network[ann], network['dist'], 10000)
            results[ann][f'assort_all_{null_type}'][i, :] = wei_assort_batch(
                A, nulls, n_batch=100, directed=directed)

            # compute standardized assortativity + p-values
            results[ann]['assort_all_z'][i] = standardize_scores(
                results[ann][f'assort_all_{null_type}'][i, :],
                results[ann]['assort_all'][i])
            results[ann]['assort_all_p'][i] = get_p_value(
                results[ann][f'assort_all_{null_type}'][i, :],
                results[ann]['assort_all'][i])

    return results


def compute_heterophilic_assortativity(network, attributes, permutations,
                                       attribute_names):
    '''
    Function to compute the heterophilic assortativity between a list of
    annotations.

    Parameters:
    ----------
    network: dict
        Dictionary storing relevant information about the network
    attributes: (n_nodes, n_attributes) ndarray
        Array storing annotation scores for a list of attributes. Each row in
        the matrix correspond to a single attribute.
    permutations: (n_nodes, n_permutations) ndarray
        Array storing the permutations (spun) used to compute the standardized
        heterophilic assortativity scores.
    attribute_names: list
        Names of each attribute. The index of an attribute's name indicate
        the row to which it is associated in the `attributes` array.

    Returns:
    -------
    results: dict
        Dictionary storing the results computed with this function. They
        include:
        `names`: Names of each attribute
        `assort`: Mixing matrix storing the assortativity score for
            each pair of annotation.
        `assort_spin`: Mixing matrices storing the assortativity score for
            each each pair of annotation in permuted connectomes.
        `assort_p`: Matrix of p-values associated with each assortativity
            score.
        `assort_z`: Mixing matrix of standardized assortativity scores for
            each pair of annotation.
    '''

    n_attributes, _ = attributes.shape
    _, n_perm = permutations.shape

    A = network['adj']

    results = {}
    results['names'] = attribute_names
    results['assort'] = np.zeros((n_attributes, n_attributes))
    results['assort_spin'] = np.zeros((n_perm, n_attributes, n_attributes))
    results['assort_p'] = np.zeros((n_attributes, n_attributes))
    results['assort_z'] = np.zeros((n_attributes, n_attributes))

    for i in trange(n_attributes):
        for j in trange(n_attributes):

            M = attributes[i, :]
            N = attributes[j, :]

            # empirical assortativity
            results['assort'][i, j] = weighted_assort(A, M, N)

            # Spun assortativity
            for k in range(n_perm):
                perm_k = permutations[:, k]
                A_perm = A[perm_k, :][:, perm_k]
                results['assort_spin'][k, i, j] = weighted_assort(A_perm, M, N)

            # compute significance
            results['assort_p'][i, j] = get_p_value(
                results['assort_spin'][:, i, j], results['assort'][i, j])
            # compute standardized scores
            results['assort_z'][i, j] = standardize_scores(
                results['assort_spin'][:, i, j], results['assort'][i, j])

    return results


def compute_homophilic_ratio(network, annotations):
    '''
    Function to compute the homophilic ratio of annotations in a network

    Parameters
    ----------
    network: dict
        Dictionary storing relevant information about the network
    annotations: list
        List of annotation names. These names should correspond to keys in the
        network dictionary.

    return
    -------
    results: dict
        Dictionary storing homophilic ratio of each annotation in the network.
    '''
    A = network['adj'].copy()

    results = {}
    for ann in annotations:

        M_diff = edge_diff(network[ann])
        nodal_mean_diff = np.average(M_diff, weights=A, axis=0)
        results[ann] = nodal_mean_diff / np.mean(M_diff, axis=0)

    return results


def edge_diff(M, N=None):
    '''
    Function to compute the absolute differences in the annotations of
    individual edges in a network.

    Parameters:
    ----------
    M: (n,) ndarray
        Vector of annotation scores
    N: (n,) ndarray
        Vector of annotation scores

    Returns:
    ----------
    diff: (n, n) ndarray
        Matrix of pairwise absolute differences between the annotation scores
        of pair of nodes.
    '''

    n_nodes = len(M)
    M_in = np.repeat(M[:, np.newaxis], n_nodes, axis=1)
    if N is None:
        M_out = M_in.T
    else:
        M_out = np.repeat(N[np.newaxis, :], n_nodes, axis=0)

    diff = np.abs(M_in - M_out)

    return diff
