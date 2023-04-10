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
from matplotlib.colors import is_color_like, ListedColormap, to_rgba
from palettable.colorbrewer.diverging import Spectral_11_r, RdBu_11_r
from palettable.colorbrewer.sequential import Reds_3, Blues_3, GnBu_9
from palettable.cartocolors.sequential import SunsetDark_7
from scipy.stats import pearsonr, zscore
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from itertools import repeat, chain, combinations
from netneurotools.plotting import plot_fsaverage

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
    Utility function to save pickled dictionary containing the data used in
    these experiments.

    Parameters
    ----------
    data: dict
        Dictionary storing the data that we want to save.
    path: str
        path of the pickle file
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


def get_p_value(perm, emp, axis=0):
    '''
    Utility function to compute the p-value (two-tailed) of a score, relative
    to a null distribution.

    Parameters
    ----------
    perm: array-like
        Null distribution of (permuted) scores.
    emp: float or array-like
        Empirical score.
    axis: float
        Axis of the `perm` array associated with the null scores.
    '''

    k = perm.shape[axis]
    perm_moved = np.moveaxis(perm, axis, 0)
    perm_mean = np.mean(perm_moved, axis=0)

    # Compute p-value
    num = (np.count_nonzero(abs(perm_moved-perm_mean) > abs(emp-perm_mean),
                            axis=0))
    den = k
    pval = num / den

    return pval


def get_cmap(colorList):
    '''
    Function to get a colormap from a list of colors
    '''
    n = len(colorList)
    c_all = np.zeros((256, 4))
    m = int(256/(n-1))
    for i in range(n):

        if isinstance(colorList[i], str):
            color = to_rgba(colorList[i])
        else:
            color = colorList[i]

        if i == 0:
            c_all[:int(m/2)] = color
        elif i < n-1:
            c_all[((i-1)*m)+(int(m/2)):(i*m)+(int(m/2))] = color
        else:
            c_all[((i-1)*m)+(int(m/2)):] = color

    cmap = ListedColormap(c_all)

    return cmap


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


def fill_triu(A):
    '''
    Function to fill the triu indices of a matrix with the elements of the
    tril matrix
    '''

    n = len(A)
    A[np.triu_indices(n)] = A.T[np.triu_indices(n)]
    return A


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

    # Cartocolors | Sequential
    cmaps['SunsetDark_7'] = SunsetDark_7.mpl_colormap

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
                          sig_colors, figsize=None, barwidth=0.5,
                          ylim=None, tight_layout=True):
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
    figsize: tuple
        Tuple specifying the width and height of the figure in dots-per-inch.
    barwidth: float
        Width of the bars.
    ylim: float
        Y-limit.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    n_annotations = len(results)

    if figsize is None:
        figsize = (n_annotations/2, 3)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i, ann in enumerate(results):

        if ann['assort_p_fdr'] < 0.05:
            color = sig_colors[i]
        else:
            color = non_sig_colors[i]

        plt.bar(i, ann['assort_z'], width=barwidth, color=color,
                edgecolor=color, zorder=1)
    plt.plot([0, n_annotations], [0, 0], color='lightgray', linestyle='dashed',
             zorder=0)
    ax.set_ylabel("z-assortativity")
    ax.set_xlabel("annotation")
    ax.set_xticks(np.arange(len(results)), labels=annotation_labels,
                  rotation='vertical')
    if ylim is not None:
        margin = 0.05 * (ylim[1] - ylim[0])
        ax.set_ylim(bottom=ylim[0]-margin, top=ylim[1]+margin)
    if tight_layout:
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

        assort_p_fdr = results[key]['assort_all_p_fdr']
        assort_z = results[key]['assort_all_z']

        # Set color
        color = np.zeros((n_box), dtype='object')
        color[:] = non_sig_colors[i]
        color[assort_p_fdr < 0.05] = sig_colors[i]

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


def plot_regression(X, Y, x_label=None, y_label=None, s=5, figsize=(3, 3),
                    alpha=0.5, permutations=None):
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
        Label of the y-axis.
    s: float
        Size of the markers in the scatterplot.
    figsize: tuple of floats
        Size of the matplotlib figure.
    alpha: float
        Transparancy of the markers in the scatterplot
    permutations: (n, n_perm) ndarray
        Permutations used to compute the significance of the relatiosnhip.

    Returns:
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    '''

    r_results = pearsonr(X, Y)
    r, p_perm = r_results
    CI = r_results.confidence_interval()
    df = len(X) - 2

    if permutations is not None:
        p_spin = get_corr_spin_p(X, Y, permutations)
    fig = plt.figure(figsize=(3, 3))
    sns.regplot(
        x=X, y=Y, color='black', truncate=False,
        scatter_kws={'s': 5, 'rasterized': True,
                     'alpha': alpha, 'edgecolor': 'none'}
                )
    plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if permutations is not None:
        p = p_spin
        plt.title(f"r={r:.2f}; p_spin={p:.4f}")
    else:
        p = p_perm
        if p < 0.00001:
            plt.title(f"r={r:.2f}; p_perm={p:.5e}")
        else:
            plt.title(f"r={r:.2f}; p_perm={p:.5f}")
    plt.tight_layout()

    return fig, (r, p, df, CI)


def plot_heatmap(values, xlabels, ylabels, cbarlabel="values",
                 cmap="viridis", vmin=None, vmax=None, grid_width=3,
                 figsize=None, text_size=12, sigs=None, text=False,
                 tight_layout=True):
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
    text: bool
        Boolean indicating whether we want the values plotted as text on top
        of the heatmap.
    tight_layout: bool
        Boolean indicating whether we want a tight layout for the figure
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
    # Add text
    if text:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                im.axes.text(j, i, "{:.2f}".format(values[i, j]),
                             horizontalalignment='center',
                             verticalalignment='center', color="black",
                             fontsize=text_size)

    if tight_layout:
        fig.tight_layout()

    return fig


def plot_homophilic_ratios(ratios, ann, coords, lhannot, rhannot,
                           noplot, order, vmin, vmax, hemi='', n_nodes=None):

    # If hemi is only left hemisphere, set the values on the right to the mean
    if hemi == "L":
        scores = np.zeros((n_nodes))+np.mean(ratios)
        if order == 'RL':
            scores[(n_nodes-len(ratios)):] = ratios
            ratios = scores
        elif order == 'LR':
            scores[:len(ratios)] = ratios
    else:
        scores = ratios

    # plot homophilic ratios on brain surface
    surface_image = plot_fsaverage(
        scores, lhannot=lhannot, rhannot=rhannot, noplot=noplot, order=order,
        views=['lateral', 'm'], vmin=vmin, vmax=vmax,
        colormap=GnBu_9.mpl_colormap,
        data_kws={'representation': 'wireframe', 'line_width': 4.0})

    # plot homophilic ratios on dotted brain
    size_change = abs(zscore(ratios))
    size_change[size_change > 5] = 5
    size = 40 + (10 * size_change)
    dot_image, _ = plot_network(
        None, coords[:, :2], None, ratios, s=size,
        view_edge=False, node_cmap=GnBu_9.mpl_colormap, node_vmin=vmin,
        node_vmax=vmax)

    return surface_image, dot_image


def plot_brain_surface(scores, lhannot, rhannot, noplot, order, colormap,
                       vmin=None, vmax=None):
    '''wrapper function to call plot_fsaverage more efficiently'''
    data_kws = {'representation': 'wireframe', 'line_width': 4.0}
    if vmin is None:
        vmin = scores.min()
    if vmax is None:
        vmax = scores.max()

    return plot_fsaverage(scores, lhannot=lhannot, rhannot=rhannot,
                          noplot=noplot, order=order, colormap=colormap,
                          views=['lateral', 'm'], data_kws=data_kws,
                          vmin=vmin, vmax=vmax)


def plot_SC_FC_heterophilic_comparison(SC_receptors, FC_receptors, SC_layers,
                                       FC_layers):
    '''
    Function to plot the scatterplot comparing heterophilic mixing in the
    structural and functional connectomes. Data points that are significant
    in FC are colored in red if they are positive, and in blue if they are
    negative.

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
            x=X['a_z'], y=Y['a_z'],
            color='lightgray', s=8, rasterized=True)
        pos_sig_FC = (Y['a_p_fdr'] < 0.05) & (Y['a_z'] > 0)
        plt.scatter(x=X['a_z'][pos_sig_FC],
                    y=Y['a_z'][pos_sig_FC],
                    color='#67001F',  s=8, rasterized=True)
        neg_sig_FC = (Y['a_p_fdr'] < 0.05) & (Y['a_z'] < 0)
        plt.scatter(x=X['a_z'][neg_sig_FC],
                    y=Y['a_z'][neg_sig_FC],
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
        (SC_receptors['a_z'].flatten(),
         SC_layers['a_z'].flatten()),
        axis=0)
    FC_a_z_all = np.concatenate(
        (FC_receptors['a_z'].flatten(),
         FC_layers['a_z'].flatten()),
        axis=0)
    r, _ = pearsonr(SC_a_z_all, FC_a_z_all)
    plt.title(f"r={r:.2f}")

    return fig


def plot_PC1_assortativity_correlations(PC1_r, PC1_r_prod, data, z_assort,
                                        barplot_size=(5, 2), grid_width=0.25):
    '''
    Function to plot results exploring the relationship between assortativity
    and correlations with PC1 (SUPPLEMENTARY FIGURE 6)

    Parameters
    ----------
    PC1_r: (m,) ndarray
        Correlations between each brain map and the first component of the
        network.
    PC1_r_prod: (m, m) ndarray
        Product of the correlations stored in PC1_r
    data: dict
        Dictionary storing either the laminar thickness or the receptor density
        data.
    z_assort: (m, m) ndarray
        Z-assortativity (i.e. heterophilic mixing) of each pair of brain map.
    barplot_size: tuple of floats
        Size of the barplot figure.
    grid_width: float
        Width of the grid in the r-product heatmap figure.

    Returns
    -------
    fig1: maptlotlib figure
        Barplot of the correlations between brain maps and PC1.
    fig2: matplotlib figure
        Heatmap of the r-product for each pair of brain map.
    fig3: matplotlib figure
        Scatterplot of the relationship between r-product and z-assortativity.
    reg: tuple
        Tuple storing statistics of the regression plotted in fig3. This tuple
        contains: (r, p, df, CI).
    '''

    labels = data['names']
    n_annotations = len(PC1_r)

    # plot barplot of correlations
    order = np.argsort(PC1_r)
    fig1 = plt.figure(figsize=barplot_size)
    plt.bar(np.arange(n_annotations), PC1_r[order], edgecolor='black',
            color='lightgray')
    plt.xticks(np.arange(n_annotations),
               labels=np.asarray(labels)[order],
               rotation=90)
    plt.xlabel("annotation")
    plt.ylabel('r')

    # Get upper triu indices of layer correlations
    r_prod_ld = PC1_r_prod[np.tril_indices(n_annotations)]
    a_z_ld = z_assort[np.tril_indices(n_annotations)]
    X = a_z_ld
    Y = r_prod_ld

    # Plot scatterplot relationship between r (product) and z-assort
    fig2, reg = plot_regression(X, Y, x_label='z-assort',
                                y_label='r (product)', s=10,
                                alpha=None, figsize=(2.5, 2.5))

    # Plot heatmap of r (products)
    m = max(abs(PC1_r_prod.min()), PC1_r_prod.max())
    fig3 = plot_heatmap(PC1_r_prod, labels, labels,
                        text=False, cmap=RdBu_11_r.mpl_colormap,
                        vmin=-m, vmax=m, grid_width=grid_width,
                        figsize=(3.4, 3.4), text_size=17)

    return fig1, fig2, fig3, reg


def boxplot(results, figsize=(2, 3), widths=0.8, showfliers=True,
            edge_colors='black', face_colors='lightgray',
            median_color=None, significants=None, positions=None, vert=True,
            ax=None):
    '''
    Function to plot results in a boxplot

    Parameters
    ----------
    results: (n_boxes, n_observations) ndarray
        Results to be plotted in the boxplot
    '''

    # Setup the flierprops dictionary
    flierprops = dict(marker='+',
                      markerfacecolor='lightgray',
                      markeredgecolor='lightgray')

    # Initialize the figure (if no `ax` provided)
    if ax is None:
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = plt.gca()
    else:
        fig = plt.gcf()

    n_boxes = len(results)

    if positions is None:
        positions = np.arange(1, n_boxes + 1)

    if is_color_like(edge_colors):
        edge_colors = [edge_colors] * n_boxes
    if is_color_like(face_colors):
        face_colors = [face_colors] * n_boxes

    # Plot each box individually
    for i in range(n_boxes):

        bplot = ax.boxplot(results[i],
                           widths=widths,
                           showfliers=showfliers,
                           patch_artist=True,
                           zorder=0,
                           flierprops=flierprops,
                           showcaps=False,
                           vert=vert,
                           positions=[positions[i]])

        for element in ['boxes', 'whiskers', 'fliers',
                        'means', 'medians', 'caps']:
            if element == 'medians' and median_color is not None:
                plt.setp(bplot[element], color=median_color)
            else:
                plt.setp(bplot[element], color=edge_colors[i])

        for patch in bplot['boxes']:

            if significants is not None:
                if significants[i]:
                    patch.set(facecolor=face_colors[i])
                else:
                    patch.set(facecolor='white')
            else:
                patch.set(facecolor=face_colors[i])

    return fig


'''
RESULTS FUNCTIONS
'''


def generate_moran_nulls(scores, network, n_nulls, species='non_human',
                         hemiid=None):
    '''
    Function to generate Moran nulls. This function relies on the brainspace
    toolbox.

    Parameters
    ----------
    scores: (n,) ndarray
        Annotation scores for which we want to generate a null distribution
    network: dict
        Dictionary storing relevant information about the network
    n_nulls: int
        Number of null annotations to generate.
    species: str
        The species for which we want to generate Moran nulls. If `human`, then
        nulls preserve the homotopy across hemispheres and are computed using
        the geodesic distance between parcels.
    hemiid: (n,) ndarray
        Label, for each parcel, indicating whether it is located in the left or
        the right hemisphere. Used when the species is `human`.

    Returns
    -------
    nulls: (n_nulls, n) ndarray
        Array of null annotations

    '''

    if species == 'non_human':
        dist = network['dist']
    elif species == 'human':
        dist = [network['geo_dist_L'], network['geo_dist_R']]

    w = spatial_weights(dist, species=species, hemiid=hemiid)

    rand_seed = np.random.default_rng().integers(0, 2**32)

    if species == 'non_human':
        moranRandom = MoranRandomization(tol=1e-5, n_rep=n_nulls,
                                         random_state=rand_seed)
        moranRandom.fit(w)
        nulls = moranRandom.randomize(scores)

    elif species == 'human':

        right_id, left_id = hemiid == 'R', hemiid == 'L'
        right_scores, left_scores = scores[right_id], scores[left_id]

        # Generate nulls for right hemisphere
        w_right = w[right_id, :][:, right_id]
        moran_right = MoranRandomization(tol=1e-5, n_rep=n_nulls,
                                         random_state=rand_seed)
        moran_right.fit(w_right)
        nulls_right = moran_right.randomize(right_scores)

        # Generate nulls for left hemisphere
        w_left = w[left_id, :][:, left_id]
        moran_left = MoranRandomization(tol=1e-5, n_rep=n_nulls,
                                        random_state=rand_seed)
        moran_left.fit(w_left)
        nulls_left = moran_left.randomize(left_scores)

        # Concatenate the two
        n_nodes = len(scores)
        nulls = np.zeros((n_nulls, n_nodes))
        nulls[:, right_id] = nulls_right
        nulls[:, left_id] = nulls_left

    return nulls


def spatial_weights(dist, species='non_human', hemiid=None):
    '''
    Function to get the spatial weight matrix of our network used to generate
    moran spatial nulls.

    Parameters
    ----------
    dist: (n, n) ndarray or tuple
        Matrix storing the euclidean distances between each node in the
        network. If a tuple, it must represent the geodesic distance between
        each parcel, separately for each connectome. With the left hemisphere
        are the first entry in the tuple, followed by the right hemisphere.
    species: str
        The species for which we want to generate Moran nulls. If `human`, then
        spatial weights are computed using the geodesic distance between
        parcels.
    hemiid: (n,) ndarray
        Label, for each parcel, indicating whether it is located in the left or
        the right hemisphere. Used when the species is `human`.

    Returns
    -------
    w: (n, n) ndarray
        Matrix storing the spatial weights for each pair of node in the
        network.
    '''

    if species == 'non_human':
        w = dist.copy()

    elif species == 'human':
        w = np.zeros((len(hemiid), len(hemiid)))
        right_id, left_id = hemiid == 'R', hemiid == 'L'
        w[np.ix_(left_id, left_id)] = dist[0]
        w[np.ix_(right_id, right_id)] = dist[1]

    w[w > 0] = 1/w[w > 0]
    np.fill_diagonal(w, 1)

    return w


def compute_standardized_assortativity(network, null_type, annotations,
                                       directed=True, moran_kwargs=None,
                                       species=None):
    '''
    Function to compute the standardized assortativity of a list of annotation
    for a given networks.

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
        directed, setting this parameter to `False` will increase the speed of
        the computations.
    moran_kwargs: dict
        Keyword arguments passed to the `generate_moran_nulls` function,
        specifying the species for which we want to generate the null
        distribution and the hemiids (if species is `human`).
    species: str
        Denotes the species associated with the data. This must be specified
        when the null_type is `burt`.

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

    if moran_kwargs is None:
        moran_kwargs = {}

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
            nulls = generate_moran_nulls(network[ann], network, 10000,
                                         **moran_kwargs)
        elif null_type == 'burt':
            nulls = np.load(f"data/burt_nulls/{species}/{ann}.npy")
        results[ann][f'assort_{null_type}'] = wei_assort_batch(
            network['adj'], nulls, n_batch=100, directed=directed)

        # compute standardized assortativity + p-values
        results[ann]['assort_z'] = standardize_scores(
            results[ann][f'assort_{null_type}'], results[ann]['assort'])
        results[ann]['assort_p'] = get_p_value(
            results[ann][f'assort_{null_type}'], results[ann]['assort'])

    # compute fdr-corrected p-values
    p_vals = [results[ann]['assort_p'] for ann in annotations]
    _, p_fdr, _, _ = multipletests(p_vals, method='fdr_by')
    for n, ann in enumerate(annotations):
        results[ann]['assort_p_fdr'] = p_fdr[n]

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
        results[ann]['assort_all_p_fdr'] = np.zeros((n_bins))

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
                nulls = generate_moran_nulls(network[ann], network, 10000)
            results[ann][f'assort_all_{null_type}'][i, :] = wei_assort_batch(
                A, nulls, n_batch=100, directed=directed)

            # compute standardized assortativity + p-values
            results[ann]['assort_all_z'][i] = standardize_scores(
                results[ann][f'assort_all_{null_type}'][i, :],
                results[ann]['assort_all'][i])
            results[ann]['assort_all_p'][i] = get_p_value(
                results[ann][f'assort_all_{null_type}'][i, :],
                results[ann]['assort_all'][i])

    # compute FDR-corrected p-values
    for i, percent in enumerate(tqdm(percent_kept)):
        p_values = [results[ann]['assort_all_p'][i] for ann in annotations]
        _, p_fdr, _, _ = multipletests(p_values, method='fdr_by')
        for n, ann in enumerate(annotations):
            results[ann]['assort_all_p_fdr'][i] = p_fdr[n]

    return results


def compute_heterophilic_assortativity(network, attributes, permutations,
                                       attribute_names, mask=None):
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
    mask: (n_attributes, n_attributes) ndarray
        Boolean array specifying the pair of attributes for which we want
        to compute their heterophilic assortativity.

    Returns:
    -------
    results: dict
        Dictionary storing the results computed with this function. They
        include:
        `names`: Names of each attribute
        `a`: Mixing matrix storing the assortativity score for
            each pair of annotation.
        `a_spin`: Mixing matrices storing the assortativity score for
            each each pair of annotation in permuted connectomes.
        `a_p`: Matrix of p-values associated with each assortativity
            score.
        `a_z`: Mixing matrix of standardized assortativity scores for
            each pair of annotation.
    '''

    n_attributes, _ = attributes.shape
    _, n_perm = permutations.shape

    A = network['adj']

    if mask is None:
        mask = np.ones((n_attributes, n_attributes), dtype=bool)

    results = {}
    results['names'] = attribute_names
    results['a'] = np.zeros((n_attributes, n_attributes))
    results['a_spin'] = np.zeros((n_perm, n_attributes, n_attributes))
    results['a_p'] = np.zeros((n_attributes, n_attributes))
    results['a_z'] = np.zeros((n_attributes, n_attributes))

    # spun assortativity (change n_jobs param to run in parallel)
    permutations = permutations.T.tolist()
    results['a_spin'] = np.asarray(
        Parallel(n_jobs=1)(
            delayed(_compute_spun_hetero_assort)(A, perm, attributes, mask)
            for A, perm, attributes, mask
            in tqdm(zip(repeat(A, n_perm),
                        permutations,
                        repeat(attributes, n_perm),
                        repeat(mask, n_perm)),
                    total=n_perm)
            )
        )

    for i in range(n_attributes):
        for j in range(n_attributes):
            M = attributes[i, :]
            N = attributes[j, :]
            if mask[i, j]:
                # empirical assortativity
                results['a'][i, j] = weighted_assort(A, M, N)
                # significance
                results['a_p'][i, j] = get_p_value(
                    results['a_spin'][:, i, j], results['a'][i, j])
                # standardized scores
                results['a_z'][i, j] = standardize_scores(
                    results['a_spin'][:, i, j], results['a'][i, j])
            else:
                results['a'][i, j] = np.nan
                results['a_p'][i, j] = np.nan
                results['a_z'][i, j] = np.nan

    # Get FDR-corrected p-values
    _, p_fdr, _, _ = multipletests(
        results['a_p'][np.tril_indices(n_attributes)], method='fdr_by')
    results['a_p_fdr'] = np.zeros(results['a_p'].shape)
    results['a_p_fdr'][:] = np.nan
    results['a_p_fdr'][np.tril_indices(n_attributes)] = p_fdr

    return results


def _compute_spun_hetero_assort(A, perm, attributes, mask):
    '''
    Function used to compute heterophilic assortativity, in parallel
    '''
    n_attributes = len(attributes)
    results = np.zeros((n_attributes, n_attributes))
    A_perm = A[perm, :][:, perm]
    for i in range(n_attributes):
        for j in range(n_attributes):
            M = attributes[i, :]
            N = attributes[j, :]
            if mask[i, j]:
                results[i, j] = weighted_assort(A_perm, M, N)
            else:
                results[i, j] = np.nan

    return results


def compute_homophilic_ratio(network, annotations):
    '''
    Function to compute the homophilic ratio of annotations in a network

    Parameters
    ----------
    network: dict
        Dictionary storing relevant information about the network.
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


def compute_partial_assortativity_all(network, annotations):
    '''
    Function to compute the partial assortativity of a list of annotations.

    Parameters
    ----------
    network: dict
        Dictionary storing relevant information about the network.
    annotations: list
        List of annotation names. These names should correspond to keys in the
        network dictionary.

    Return
    -------
    results: dict
        Dictionary storing the partial assortativity of each annotation,
        relative to the other annotations.
    '''

    # Initialize results dictionary
    results = {}
    n_keys = len(annotations)
    results['keys'] = annotations
    results['r'] = np.zeros((n_keys, n_keys))
    results['r_perm'] = np.zeros((n_keys, n_keys, 1000))

    # Setup important variables
    A = network['adj'].copy()
    edges = A > 0
    edge_weights = A[edges]

    # For each combination of annotation pair
    for i, M in enumerate(annotations):
        M_scores = zscore(network[M])
        for j, N in enumerate(annotations):
            N_scores = zscore(network[N])

            # Compute results (empirical)
            results['r'][i, j], Y_res_in, Y_res_out = partial_assortativity(
                A, M_scores, N_scores)

            # Compute results (permuted annotations)
            if i == j:
                results['r_perm'][i, j, :] = np.nan
            else:
                for k in trange(1000):
                    results['r_perm'][i, j, k] = weighted_correlation(
                        np.random.permutation(Y_res_in[:, 0]),
                        Y_res_out[:, 0],
                        edge_weights)

    # Get p-values
    results['p'] = get_p_value(results['r_perm'], results['r'], axis=2)
    results['p_fdr'] = np.zeros(results['p'].shape)
    for i in range(n_keys):
        non_nan_ids = ~np.isnan(results['r_perm'][i, :, 0])
        _, results['p_fdr'][i, non_nan_ids], _, _ = multipletests(
            results['p'][i, non_nan_ids], method='fdr_by')

    return results


def correlate_with_PC(brain_maps, PC1):
    '''
    Function to compute the correlations between brain maps and the principal
    component of network connetivity of a given network. This function also
    returns the product of the correlation between each pair of brain maps.

    Parameters
    ----------
    brain_maps: (n, m) ndarray
        Brain maps. `n` denotes the number of nodes in each parcellated brain
        maps. `m` denotes the total number of brain maps.
    PC1: (n,) ndarray
        First component of the adjacency matrix of the network.

    Returns:

    '''

    _, n_maps = brain_maps.shape
    r = np.zeros((n_maps))
    r_prod = np.zeros((n_maps, n_maps))

    for i in range(n_maps):
        r[i], _ = pearsonr(PC1, brain_maps[:, i])

    for i in range(n_maps):
        for j in range(n_maps):
            r_prod[i, j] = r[i] * r[j]

    return r, r_prod


def partial_assortativity(A, M, N):
    '''
    Function to compute the assortativity of annotation `M`, with the
    annotation scores of `N` regressed out.

    Parameters
    ----------
    A: (n, n) ndarray
        Adjacency matrix of the network
    M: (n,) ndarray
        Vector of annotation scores for annotation `M`
    N: (n,) ndarray
        Vector of annotation scores for annotation `N`

    Returns
    -------
    r: float
        Partial assortativity
    Y_res_in: (m,) ndarray
        Residuals of the regression between N(in) and M(in)
    Y_res_out: (m,) ndarray
        Residuals of the regression between N(in) and M(out)
    '''

    def fit_model(X, Y, W):
        ''' Fit a weighted linear model between two variables '''
        reg = LinearRegression().fit(X, Y,  sample_weight=W)
        Y_res = Y - reg.predict(X)  # residuals
        return reg, Y_res

    # Get info about the network
    n_nodes = len(A)
    edges = A > 0
    edge_weights = A[edges]
    edge_idx = np.flatnonzero(A)

    # Get in- and out- annotations
    N_in = N[np.floor_divide(edge_idx, n_nodes)]
    M_in = M[np.floor_divide(edge_idx, n_nodes)]
    M_out = M[np.mod(edge_idx, n_nodes)]

    # Get residuals for in- and out- annotations
    _, Y_res_out = fit_model(N_in[:, np.newaxis],
                             M_out[:, np.newaxis],
                             edge_weights)
    _, Y_res_in = fit_model(N_in[:, np.newaxis],
                            M_in[:, np.newaxis],
                            edge_weights)

    # Compute correlations for these residuals
    r = weighted_correlation(Y_res_in[:, 0], Y_res_out[:, 0], edge_weights)

    return r, Y_res_in, Y_res_out


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


def weighted_correlation(x, y, w):
    '''
    Function to compute the weighted Pearsonr correlation between two
    variables.

    Parameters
    ----------
    x: (n,) ndarray
        Independent variable
    y: (n,) ndarray
        Dependent variable
    w: (n,) ndarray
        Weights

    Returns:
    -------
    r: float
        Weighted correlation

    '''

    sum_w = np.sum(w)

    x_bar = x - np.average(x, axis=None, weights=w)
    y_bar = y - np.average(y, axis=None, weights=w)

    cov_x_y = np.sum(w * x_bar * y_bar) / sum_w
    cov_x_x = np.sum(w * x_bar * x_bar) / sum_w
    cov_y_y = np.sum(w * y_bar * y_bar) / sum_w

    return cov_x_y / np.sqrt(cov_x_x * cov_y_y)


def get_PC1(A):
    '''
    Function to get the first principal component of an adjacency matrix

    Parameters
    ----------
    A: (n, n) ndarray
        Adjacency matrix

    Returns
    -------
    PC1: (n,) ndarray
        First principal component
    '''

    pca = PCA(n_components=10)
    pca.fit(A)
    return pca.components_[0, :]


def weighted_multi_regression_and_dominance(A, network, Y_key, X_keys):

    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    n_nodes = len(A)

    # Dependent variable are the `Y` scores at the `out` endpoints
    Y = network[Y_key]
    Y = np.repeat(Y[np.newaxis, :], n_nodes, axis=0)[A > 0]

    # Independent variables are the `X` scores at the `in` endpoints
    X = {}
    for X_key in X_keys:
        X[X_key] = network[X_key]
        X[X_key] = np.repeat(X[X_key][:, np.newaxis], n_nodes, axis=1)[A > 0]

    # Weight of each sample (strength of connection)
    weights = A[A > 0]

    # Get powersets of variables
    X_models = list(powerset(X_keys))

    # Fit each subset
    R2 = []
    for X_model in X_models:
        X_all = []
        for i in range(len(X_model)):
            X_all.append(X[X_model[i]])

        if X_model != ():
            X_all = np.array(X_all).T
            reg = LinearRegression().fit(X_all, Y, sample_weight=weights)
            R2.append(reg.score(X_all, Y, sample_weight=weights))
        else:
            R2.append(0)

    dominance = []
    # Compute relative increase in R2 for each X
    for key in X_keys:

        # Find models that use specific X as a regressor
        X_models_key_id = []
        X_models_key = []
        for i, X_model in enumerate(X_models):
            if key in X_model:
                X_models_key.append(X_model)
                X_models_key_id.append(i)

        # For each, get the same model without specific X as a regressor
        X_models_no_key_id = []
        X_models_no_key = []
        for X_model_key in X_models_key:
            model = list(X_model_key)
            model.remove(key)
            X_models_no_key_id.append(X_models.index(tuple(model)))
            X_models_no_key.append(model)

        R2_array = np.array(R2)

        R2_diff = R2_array[X_models_key_id] - R2_array[X_models_no_key_id]

        dominance.append(R2_diff.mean())

    return dominance, R2, X_models


def reorganize_dominance_results(dominance_results, keys, n_nulls):

    results = {}

    heatmap = np.array([dominance_results[key]['dominance'] for key in keys])
    percentages = heatmap / heatmap.sum(axis=1)[:, np.newaxis]

    n_keys = len(keys)

    # Compute p-values
    dominance_p = np.zeros((n_keys, n_keys))
    for n, key_n in enumerate(keys):
        for m, key_m in enumerate(keys):
            dominance_spin = [dominance_results[key_n]['dominance_spin'][i][m]
                              for i in range(n_nulls)]
            dominance = dominance_results[key_n]['dominance'][m]
            dominance_p[n, m] = get_p_value(np.array(
                dominance_spin), dominance)

    results['keys'] = keys
    results['dominance_percentage'] = percentages
    results['dominance_p'] = dominance_p

    results['R2'] = np.zeros((n_keys))
    results['R2_spin'] = np.zeros((n_keys, n_nulls))
    results['R2_p'] = np.zeros((n_keys))

    for n, key_n in enumerate(keys):
        results['R2'][n] = dominance_results[key_n]['R2'][-1]
        for k in range(n_nulls):
            results['R2_spin'][n, k] = dominance_results[key_n]['R2_spin'][k][-1]

    for n in range(len(keys)):
        results['R2_p'][n] = get_p_value(
            results['R2_spin'][n, :], results['R2'][n])

    _, results['R2_p_fdr'], _, _ = multipletests(
        results['R2_p'], method='fdr_by')

    return results


def compute_spearman_assort(network, null_type, annotations,
                            moran_kwargs=None):
    '''
    Function to compute the standardized assortativity of a list of annotations
    for a given network, using a rank-based measure of correlation between
    annotations (Spearman's rho).

    Parameters
    ----------
    network: dict
        Dictionary storing relevant information about the network
    null_type: str
        Type of null used to compute the standardized assortativity
    annotations: list
        List of annotation names. These names should correspond to keys in the
        network dictionary.
    moran_kwargs: dict
        Keyword arguments passed to the `generate_moran_nulls` function,
        specifying the species for which we want to generate the null
        distribution and the hemiids (if species is `human`).

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

    def rank_edges(M_argsort, deg, edges):

        deg_sorted = deg[M_argsort]
        deg_cumsum = np.cumsum(deg_sorted)
        M_ranked = deg_cumsum[M_argsort.argsort()][edges]

        return M_ranked

    if moran_kwargs is None:
        moran_kwargs = {}

    # Get info about the network
    A = network['adj']
    n_nodes = len(A)
    edges = A > 0
    edge_idx = np.flatnonzero(A)
    in_edges = np.floor_divide(edge_idx, n_nodes)
    in_deg = np.count_nonzero(A, axis=1)
    out_edges = np.mod(edge_idx, n_nodes)
    out_deg = np.count_nonzero(A, axis=0)

    # Results dictionary
    r = {}

    for ann in annotations:

        print(f"computing spearman's assortativity for: `{ann}`")

        r[ann] = {}

        # Load annotation
        M = network[ann]

        # Compute assortativity (empirical)
        M_argsort = M.argsort()
        M_in_ranked = rank_edges(M_argsort, in_deg, in_edges)
        M_out_ranked = rank_edges(M_argsort, out_deg, out_edges)
        r[ann]['assort'] = weighted_correlation(
            M_in_ranked, M_out_ranked, A[edges])

        # Get null distribution
        if null_type == 'spin':
            nulls = network[ann][network['spin_nulls']].T
        elif null_type == 'moran':
            nulls = generate_moran_nulls(network[ann], network, 10000,
                                         **moran_kwargs)

        r[ann][f'assort_{null_type}'] = np.zeros((10000))
        for k in trange(10000):

            # Compute assortativity (nulls)
            M_null_argsort = nulls[k, :].argsort()
            nulls_in_ranked = rank_edges(M_null_argsort, in_deg, in_edges)
            nulls_out_ranked = rank_edges(M_null_argsort, out_deg, out_edges)

            # weighted correlation
            r[ann][f'assort_{null_type}'][k] = weighted_correlation(
                nulls_in_ranked, nulls_out_ranked, A[edges])

        r[ann]['assort_z'] = standardize_scores(
            r[ann][f'assort_{null_type}'], r[ann]['assort'])
        r[ann]['assort_p'] = get_p_value(
            r[ann][f'assort_{null_type}'], r[ann]['assort'])

    # compute fdr-corrected p-values
    p_vals = [r[ann]['assort_p'] for ann in annotations]
    _, p_fdr, _, _ = multipletests(p_vals, method='fdr_by')
    for n, ann in enumerate(annotations):
        r[ann]['assort_p_fdr'] = p_fdr[n]

    return r
