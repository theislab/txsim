import scanpy as sc
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d


def jensen_shannon_distance_metrics(adata_sp: AnnData, adata_sc: AnnData, 
                              key:str='celltype', layer:str='lognorm', smooth_distributions:str='no',
                              min_number_cells:int=20,
                              pipeline_output: bool=True, show_NaN_ct:bool=False):
    """Calculate the Jensen-Shannon divergence between the two distributions:
    the spatial and dissociated single-cell data distributions. Jensen-Shannon
    is an asymmetric metric that measures the relative entropy or difference 
    in information represented by two distributions.
    ----------
    adata_sp: AnnData
        annotated ``AnnData`` object containing the spatial single-cell data
    adata_sc: AnnData
        annotated ``AnnData`` object containing the dissociated single-cell data
    key: str (default: 'celltype')
        .obs column of ``AnnData`` that contains celltype information
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric
    smooth_distributions: str (default: 'no')
        whether to smooth the distributions before calculating the metric per gene and per celltype
        'no' - no smoothing
        'moving_average' - moving average
        'rolling_median' - rolling median
        'gaussian' - gaussian filter
    min_number_cells: int (default: 20)
        minimum number of cells belonging to a cluster to consider it in the analysis
    pipeline_output: bool (default: True)
        whether to return only the overall metric (pipeline style)
        (if False, will return the overall metric, per-gene metric and per-celltype metric)
    show_NaN_ct: bool (default: True)
        whether to show the cell types with NaN values (cell types which were filtered out because of the min_number_cells threshold)
    

    Returns
    -------
    overall_metric: float
        overall Jensen-Shannon divergence between the two distributions
    per_gene_metric: float
        per gene Jensen-Shannon divergence between the two distributions
    per_celltype_metric: float
        per celltype Jensen-Shannon divergence between the two distributions
    """

    ### SET UP 

    # set the .X layer of each of the adatas according to the layer parameter
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # take the intersection of genes present in adata_sp and adata_sc, as a list
    intersect_genes = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))
    intersect_celltypes = list(set(adata_sp.obs[key]).intersection(set(adata_sc.obs[key])))

    # number of genes
    n_genes = len(intersect_genes)

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect_genes]
    adata_sp=adata_sp[:,intersect_genes]

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer]= a.layers[layer].toarray()

    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[intersect_celltypes]
    celltype_count_sp = adata_sp.obs[key].value_counts().loc[intersect_celltypes]
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    # celltypes that we will use for the metric (are in both datasets and have enough cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()
    # number of celltypes we will use for the metric (are in both datasets and have enough cells)
    n_celltypes = len(intersect_celltypes)

    # subset adata_sc and adata_sp to only include eligible celltypes
    adata_sc=adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp=adata_sp[adata_sp.obs[key].isin(celltypes)]

    ################
    # OVERALL METRIC
    ################
    overall_metric = 0 
    for gene in adata_sc.var_names:
        sum = 0
        for celltype in set(adata_sp.obs['celltype']):
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype, smooth_distributions)
        overall_metric += sum
    overall_metric = overall_metric / (n_celltypes * n_genes)

    if pipeline_output: # the execution stops here if pipeline_output=True
         return overall_metric

    ################
    # PER-GENE METRIC
    ################
    per_gene_metric = pd.DataFrame(columns=['Gene', 'JSD']) 
    for gene in adata_sc.var_names:
        sum = 0
        for celltype in celltypes:
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype, smooth_distributions)
        jsd = sum / n_celltypes
        new_entry = pd.DataFrame([[gene, jsd]],
                   columns=['Gene', 'JSD'])
        per_gene_metric = pd.concat([per_gene_metric, new_entry])
    # set gene as index
    per_gene_metric.set_index('Gene', inplace=True)    
    
    ################
    # PER-CELLTYPE METRIC
    ################
    per_celltype_metric = pd.DataFrame(columns=['celltype', 'JSD'])
    for celltype in celltypes:
        sum = 0
        for gene in intersect_genes:
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype, smooth_distributions)
        jsd = sum / n_genes
        new_entry = pd.DataFrame([[celltype, jsd]],
                     columns=['celltype', 'JSD'])
        per_celltype_metric = pd.concat([per_celltype_metric, new_entry])

    # add the rows with celltypes which were filtered out because of the min_number_cells threshold and set their JSD to NaN
    if show_NaN_ct:
        celltypes_with_nan = list(set(intersect_celltypes) - set(celltypes))
        for celltype in celltypes_with_nan:
            new_entry = pd.DataFrame([[celltype, np.nan]],
                        columns=['celltype', 'JSD'])
            per_celltype_metric = pd.concat([per_celltype_metric, new_entry])
        # set celltype as index
        per_celltype_metric.set_index('celltype', inplace=True)
    ################
    return overall_metric, per_gene_metric, per_celltype_metric

def jensen_shannon_distance_per_gene_and_celltype(adata_sp:AnnData, adata_sc:AnnData, gene:str, celltype:str, smooth_distributions):
    """Calculate the Jensen-Shannon distance between two distributions:
    1. expression values for a given gene in a given celltype from spatial data
    2. expression values for a given gene in a given celltype from single-cell data
    ----------
    adata_sp: AnnData
        annotated ``AnnData`` object containing the spatial single-cell data
    adata_sc: AnnData
        annotated ``AnnData`` object containing the dissociated single-cell data
    gene: str
        gene of interest
    celltype: str
        celltype of interest
        
    Returns
    -------
    jsd: float
        Jensen-Shannon distance between the two distributions, single value
    """
    # 0. get all expression values for the gene of interest for the celltype of interest
    sp = np.array(adata_sp[adata_sp.obs['celltype']==celltype][:,gene].X)
    sc = np.array(adata_sc[adata_sc.obs['celltype']==celltype][:,gene].X)
    # 1.flatten the arrays to 1-dim vectors
    sp = sp.flatten()
    sc = sc.flatten()
    # 2. get the probability distributions for the two vectors
    P, Q = get_probability_distributions_for_sp_and_sc(sp, sc, smooth_distributions)
    return distance.jensenshannon(P, Q, base=2)


def get_probability_distributions_for_sp_and_sc(v_sp:np.array, v_sc:np.array, smooth_distributions):
    """Calculate the probability distribution vectors from one celltype and one gene
    from spatial and single-cell data
    ---------- 
    v_sp: np.array
        spatial data from one celltype and one gene, 1-dim vector
    v_sc: np.array
        single-cell data from one celltype and one gene, 1-dim vector

    Returns
    -------
    probability_distribution_sp: np.array
        probability distribution from spatial data for one celltype and one gene, 1-dim vector
    probability_distribution_sc: np.array
        probability distribution from dissociated sc data for one celltype and one gene, 1-dim vector
    """
    # 0. find the maximum and minimus value in the two given vectors
    max_value = max(max(v_sp), max(v_sc))
    min_value = min(min(v_sp), min(v_sc))

    # 1. Calculate the histograms for the two vectors
    # 1.1 spatial
    hist_sp, bin_edges = np.histogram(v_sp, bins=min(100, int(max_value - min_value + 1)), density=True)

    # 1.2 dissociated sc
    hist_sc, bin_edges = np.histogram(v_sc, bins=min(100, int(max_value - min_value + 1)), density=True)

    # 2. Smooth the distributions if the method is specified
    match smooth_distributions:
        case 'no':
            pass
        case 'moving_average':
            hist_sp = moving_average_smooth(hist_sp)
            hist_sc = moving_average_smooth(hist_sc)
        case 'rolling_median':
            hist_sp = rolling_median(hist_sp)
            hist_sc = rolling_median(hist_sc)
        case 'gaussian':
            hist_sp = gaussian_smooth(hist_sp)
            hist_sc = gaussian_smooth(hist_sc)
        case _:
            raise ValueError(f"Unknown smoothing method: {smooth_distributions}")
    return hist_sp, hist_sc

def moving_average_smooth(histogram, window_size=3):
    # Applying moving average
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_values = np.convolve(histogram, weights, 'valid')
    return np.concatenate((smoothed_values, np.zeros(window_size-1)))

def rolling_median(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def gaussian_smooth(data, sigma=1):
    smoothed_values = gaussian_filter1d(data, sigma=sigma)
    return smoothed_values


# ONGOING
# TODO: "FutureWarning: 
# The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. 
# In a future version, this will no longer exclude empty or all-NA columns when 
# determining the result dtypes. To retain the old behavior, exclude the relevant 
# entries before the concat operation.
# per_celltype_metric = pd.concat([per_celltype_metric, new_entry])"
# TODO: change the functions structure so that it is not calculating per gene and per celltype metrics twice
# TODO: my code has a match statement, so we need python >= 3.10, is that ok?
# FUTURE
# TODO: allow setting the window size or sigma for smoothing
# TODO: maybe implement Wasserstein or maybe even the Cramer distance in addition to Jensen-Shannon
####
