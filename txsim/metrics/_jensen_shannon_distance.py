import scanpy as sc
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d
from ._util import check_crop_exists
from ._util import get_bin_edges
from ._util import get_eligible_celltypes



def jensen_shannon_distance(adata_sp: AnnData, adata_sc: AnnData, 
                              key:str='celltype', layer:str='lognorm', smooth_distributions:str='no',
                              min_number_cells:int=10,
                              pipeline_output: bool=True):
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
    celltypes, adata_sc, adata_sp = get_eligible_celltypes(adata_sc, adata_sp, key=key, 
                                                           layer=layer, 
                                                           min_number_cells=min_number_cells)
    n_celltypes = len(celltypes)
    n_genes = len(adata_sc.var_names)
    overall_metric = 0

    # if there are no eligible celltypes, return NaN
    if n_celltypes == 0:
        if pipeline_output:
            return np.nan
        else:
            return np.nan, np.nan, np.nan

    ################
    # PER-CELLTYPE METRIC
    ################
    per_celltype_metric = pd.DataFrame(columns=['celltype', 'JSD'])
    for celltype in celltypes:
        sum = 0
        for gene in adata_sc.var_names:
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype, smooth_distributions)
        jsd = sum / n_genes
        new_entry = pd.DataFrame([[celltype, jsd]],
                     columns=['celltype', 'JSD'])
        per_celltype_metric = pd.concat([per_celltype_metric, new_entry])
        overall_metric += sum
    
    ################
    # OVERALL METRIC
    ################
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

    return overall_metric, per_gene_metric, per_celltype_metric

def jensen_shannon_distance_local(adata_sp:AnnData, adata_sc:AnnData,
                                               x_min:int, x_max:int, y_min:int, y_max:int,
                                               image: np.ndarray, bins: int = 10,
                                               key:str='celltype', layer:str='lognorm',
                                               min_number_cells:int=10 # the minimal number of cells per celltype to be considered
                                               ):
    """Calculate the Jensen-Shannon divergence between the spatial and dissociated single-cell data distributions
    for each gene, but using only the cells in a given local area for the spatial data.

    Parameters
    ----------
    adata_sp: AnnData
        annotated ``AnnData`` object containing the spatial single-cell data
    adata_sc: AnnData
        annotated ``AnnData`` object containing the dissociated single-cell data
    x_min: int
        minimum x coordinate of the local area
    x_max: int
        maximum x coordinate of the local area
    y_min: int
        minimum y coordinate of the local area
    y_max: int
        maximum y coordinate of the local area
    image: np.ndarray
        spatial image, represented as a numpy array
    bins: int or array_like or [int, int] or [array, array] (default: 10)
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    key: str (default: 'celltype')
        .obs column of ``AnnData`` that contains celltype information
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric
    min_number_cells: int (default: 20)
        minimum number of cells belonging to a cluster to consider it in the analysis

    Returns
    -------
    gridfield_metric: pd.DataFrame
        Jensen-Shannon divergence for each segment of the gridfield

    """
    # check if the crop existis in the image
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    bins_x, bins_y = get_bin_edges(range, bins)

    # defines the size of the gridfield_metric (np.array) 
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1

    celltypes, adata_sc, adata_sp = get_eligible_celltypes(adata_sc, adata_sp, key=key, 
                                                           layer=layer, 
                                                           min_number_cells=min_number_cells)


    #initialize the np.array that will hold the metric for each segment of the gridfield
    gridfield_metric = np.zeros((n_bins_x, n_bins_y))

    i, j = 0, 0
    for x_start, x_end in zip(bins_x[:-1], bins_x[1:]):
        i = 0
        for y_start, y_end in zip(bins_y[:-1], bins_y[1:]):    
            # instead of dataframe, take the cropped adata_sp for one bin here

            adata_sp_local = adata_sp[(adata_sp.obs['centroid_x'] >= x_start) & 
                                        (adata_sp.obs['centroid_x'] < x_end) &
                                        (adata_sp.obs['centroid_y'] >= y_start) &
                                        (adata_sp.obs['centroid_y'] < y_end)].copy()

            if adata_sp_local.shape[0] < min_number_cells:
                gridfield_metric[i,j] = np.nan   
                i += 1
            else:
                # pipeline output=True, so we only get the overall metric, maybe expand this to per gene and per celltype
                jsd = jensen_shannon_distance(adata_sp_local, adata_sc, key=key, 
                                          layer=layer, min_number_cells=min_number_cells, pipeline_output=True)
                gridfield_metric[i,j]  = jsd
            i += 1
        j += 1
            
    return gridfield_metric

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
    # 1. get the vectors for the two distributions
    sp = adata_sp[adata_sp.obs['celltype']==celltype][:,gene].X.ravel()
    sc = adata_sc[adata_sc.obs['celltype']==celltype][:,gene].X.ravel()

    # 2. get the probability distributions for the two vectors
    P, Q = get_probability_distributions(sp, sc, smooth_distributions)
    return distance.jensenshannon(P, Q, base=2)


def get_probability_distributions(v_sp:np.array, v_sc:np.array, smooth_distributions):
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
    # TODO: make window_size changeable
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_values = np.convolve(histogram, weights, 'valid')
    return np.concatenate((smoothed_values, np.zeros(window_size-1)))

def rolling_median(data, window_size=3):
    # TODO: make window_size changeable
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def gaussian_smooth(data, sigma=1):
    # TODO: make sigma changeable
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
# TODO: allow setting the window size or sigma for smoothing
# TODO: maybe implement Wasserstein or maybe even the Cramer distance in addition to Jensen-Shannon
####
