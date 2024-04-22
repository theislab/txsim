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
from scipy.optimize import curve_fit
import warnings
import os

# Ignore specific FutureWarning from pandas # TODO: remove this line after
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

def jensen_shannon_distance(adata_sc: AnnData, 
                            adata_sp: AnnData, 
                            key:str='celltype', layer:str='lognorm', 
                            min_number_cells:int=10,
                            pipeline_output: bool=True,
                            smooth_distributions:str='no',
                            window_size: int=7,
                            sigma: int=2,
                            correct_for_cell_number_dependent_decay: bool=False,
                            filter_out_double_zero_distributions: bool=True,
                            decay_csv_enclosing_folder='output',
                            initial_guess_pl=[1.5, -0.5, -0.5]):
    """Calculate the Jensen-Shannon divergence between the two distributions:
    the spatial and dissociated single-cell data distributions. Jensen-Shannon
    is an asymmetric metric that measures the relative entropy or difference 
    in information represented by two distributions
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
        'convolution' - convolution filter, moving average 
        'gaussian' - gaussian filter
    min_number_cells: int (default: 10)
        minimum number of cells belonging to a cluster to consider it in the analysis
    pipeline_output: bool (default: True)
        whether to return only the overall metric (pipeline style)
        (if False, will return the overall metric, per-gene metric and per-celltype metric)
    window_size: int (default: 7)
        window size for the convolution filter
    sigma: int (default: 2)
        standard deviation for the gaussian filter
    correct_for_cell_number_dependent_decay: bool (default: True)
        whether to correct the metric for cell number dependent decay, cannot be used simultaneously with smoothing
    

    Returns
    -------
    overall_metric: float
        overall Jensen-Shannon divergence between the two distributions
    per_gene_metric: float
        per gene Jensen-Shannon divergence between the two distributions
    per_celltype_metric: float
        per celltype Jensen-Shannon divergence between the two distributions
    """
    # 1. Preparation
    adata_sc.X = adata_sc.layers[layer]
    adata_sp.X = adata_sp.layers[layer]
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.X = a.X.toarray()
    
    if smooth_distributions == 'gaussian':
        window_size = None
    elif smooth_distributions == 'convolution':
        sigma = None
    else:
        window_size = None
        sigma = None

    # 2. calculate and save jsd decay parameters if correct_for_cell_number_dependent_decay is True
    decay_param_df = None
    if correct_for_cell_number_dependent_decay:
        decay_csv_path = os.path.join(decay_csv_enclosing_folder, 
        f"{smooth_distributions}{sigma if sigma is not None else ''}{window_size if window_size is not None else ''}_jsd_decay_params.csv")
        if not os.path.exists(decay_csv_path):
            decay_param_df = generate_jsd_decay_data(adata_sc=adata_sc,
                                                    initial_guess_pl=initial_guess_pl,
                                                    decay_csv_path=decay_csv_path)
        else:
            decay_param_df = pd.read_csv(decay_csv_path, index_col=0)
            for col in decay_param_df.columns:
                decay_param_df[col] = decay_param_df[col].apply(parse_array)
    
    # get the eligible celltypes (at leaset min_number_cells cells per celltype, in both adata_sp and adata_sc)
    celltypes, adata_sc, adata_sp = get_eligible_celltypes(adata_sc, 
                                                           adata_sp, 
                                                           key=key, 
                                                           layer=layer, 
                                                           min_number_cells=min_number_cells)

    # if there are no eligible celltypes, return NaN
    if len(celltypes) == 0:
        if pipeline_output:
            return np.nan
        else:
            return np.nan, np.nan, np.nan

    #first, make a df to store the jsds for each gene and celltype
    jsd_df = pd.DataFrame(index=adata_sc.var_names, columns=celltypes)
    # now, loop over the genes and celltypes to fill the array
    for celltype in celltypes:
        for gene in adata_sc.var_names:
            jsd = jensen_shannon_distance_per_gene_and_celltype(adata_sc=adata_sc,
                                                                adata_sp=adata_sp,
                                                                gene=gene,
                                                                celltype=celltype,
                                                                smooth_distributions=smooth_distributions,
                                                                window_size=window_size,
                                                                sigma=sigma,
                                                                correct_for_cell_number_dependent_decay=correct_for_cell_number_dependent_decay,
                                                                filter_out_double_zero_distributions=filter_out_double_zero_distributions,
                                                                decay_param_df=decay_param_df)
            jsd_df.loc[gene, celltype] = jsd

    # OVERALL METRIC
    overall_metric = jsd_df.mean().mean()
    if pipeline_output: # the execution stops here if pipeline_output=True
        return overall_metric
    # PER-CELLTYPE METRIC
    per_celltype_metric = jsd_df.mean(axis=0).to_frame(name="JSD")
    # PER-GENE METRIC
    per_gene_metric = jsd_df.mean(axis=1).to_frame(name="JSD")

    return overall_metric, per_gene_metric, per_celltype_metric

def jensen_shannon_distance_local(adata_sc:AnnData,
                                adata_sp:AnnData, 
                                x_min:int, x_max:int, y_min:int, y_max:int,
                                image: np.ndarray, bins: int = 10,
                                key:str='celltype', layer:str='lognorm',
                                min_number_cells:int=10, # the minimal number of cells per celltype to be considered
                                smooth_distributions:str='no_smoothing',
                                window_size:int=7,
                                sigma:int=2,
                                correct_for_cell_number_dependent_decay:bool=False):
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
    smooth_distributions: str (default: 'no')
        whether to smooth the distributions before calculating the metric per gene and per celltype
        'no' - no smoothing
        'convolution' - convolution filter, moving average
        'gaussian' - gaussian filter
    window_size: int (default: 7)
        window size for the convolution filter
    sigma: int (default: 2)
        standard deviation for the gaussian filter
    correct_for_cell_number_dependent_decay: bool (default: True)
        whether to correct the metric for cell number dependent decay, cannot be used simultaneously with smoothing

    Returns
    -------
    gridfield_metric: np.array
        Jensen-Shannon divergence between the two distributions for each segment of the gridfield

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
                jsd = jensen_shannon_distance(adata_sc, 
                                        adata_sp_local, 
                                        key=key, 
                                        layer=layer, 
                                        min_number_cells=min_number_cells,
                                        smooth_distributions=smooth_distributions,
                                        window_size=window_size,
                                        sigma=sigma, 
                                        pipeline_output=True,
                                        correct_for_cell_number_dependent_decay=correct_for_cell_number_dependent_decay)
                gridfield_metric[i,j]  = jsd
            i += 1
        j += 1
            
    return gridfield_metric

def jensen_shannon_distance_per_gene_and_celltype(adata_sc:AnnData, 
                                                  adata_sp:AnnData, 
                                                  gene:str, 
                                                  celltype:str,
                                                  smooth_distributions: str,
                                                  window_size: int,
                                                  sigma: int,
                                                  correct_for_cell_number_dependent_decay: bool,
                                                  filter_out_double_zero_distributions: bool,
                                                  decay_param_df: pd.DataFrame):
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
    smooth_distributions: str
        whether to smooth the distributions before calculating the metric per gene and per celltype
        'no' - no smoothing
        'convolution' - convolution filter, moving average
        'gaussian' - gaussian filter
    correct_for_cell_number_dependent_decay: bool (default: True)
        whether to correct the metric for cell number dependent decay, cannot be used simultaneously with smoothing
    window_size: int (default: 7)
        window size for the convolution filter
    sigma: int (default: 2)
        standard deviation for the gaussian filter
        
    Returns
    -------
    jsd: float
        Jensen-Shannon distance between the two distributions, single value
    """
    # 1. get the vectors for the two distributions
    sc = adata_sc[adata_sc.obs['celltype']==celltype][:,gene].X.ravel()
    sp = adata_sp[adata_sp.obs['celltype']==celltype][:,gene].X.ravel()

    # 2. get the probability distributions for the two vectors
    P, Q = get_probability_distributions(sc, sp, smooth_distributions, window_size, sigma)

    # 3. check if the distributions are None, handle accordingly
    if P is None and Q is None:
        if filter_out_double_zero_distributions:
            return None
        else:
            return 0
    elif (P is None and Q is not None) or (P is not None and Q is None):
        return 1
    else:
        jsd = distance.jensenshannon(P, Q)

    # 4. Correct for cell number dependent decay if requested
    if correct_for_cell_number_dependent_decay:
        baseline_jsd = power_law_func(len(sp), *decay_param_df.loc[gene, celltype])
        if baseline_jsd < 0:
            baseline_jsd = 0

        jsd = jsd - baseline_jsd
        if jsd < 0:
            jsd = 0

    return jsd


def get_probability_distributions(v_sc:np.array, v_sp:np.array, smooth_distributions: str,
                                  window_size:int, sigma:int):
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
    # Check if data has no variation
    if np.max(v_sc) - np.min(v_sc) <= 1e-9 and np.max(v_sp) - np.min(v_sp) > 1e-9:
        return None, np.zeros_like(v_sp)
    elif np.max(v_sc) - np.min(v_sc) > 1e-9 and np.max(v_sp) - np.min(v_sp) <= 1e-9:
        return np.zeros_like(v_sc), None
    elif np.max(v_sc) - np.min(v_sc) <= 1e-9 and np.max(v_sp) - np.min(v_sp) <= 1e-9:
        return None, None
    # only if both vectors have variation, calculate common bins
    else:
        bins1 = freedman_diaconis(v_sc)
        bins2 = freedman_diaconis(v_sp)
        num_bins = max(max(bins1, bins2) + 1, 10)
        common_bins = np.linspace(start=min(np.min(v_sc), np.min(v_sp)), 
                                  stop=max(np.max(v_sc), np.max(v_sp)),
                                  num=num_bins)
    # ensure that common_bins have a reasonable number of bins
    if np.any(np.diff(common_bins) <= 0) or len(common_bins) < 10:
        raise ValueError("Invalid bin configuration leading to zero or negative bin width.")

    # original data without smoothing
    hist_sc, _ = np.histogram(v_sc, bins=common_bins, density=True)
    hist_sp, _ = np.histogram(v_sp, bins=common_bins, density=True)

    if smooth_distributions == 'no_smoothing':
        return hist_sp, hist_sc
    else:
        # find out which vector is smaller
        if len(v_sp) <= len(v_sc):
            hist_bigger = hist_sc
            hist_to_smooth = hist_sp
        else:   
            hist_bigger = hist_sp
            hist_to_smooth = hist_sc
        # separate the zeros
        zeros_bin = hist_to_smooth[0].copy()
        hist_to_smooth_nonzeros = hist_to_smooth.copy()
        hist_to_smooth_nonzeros[0] = 0
        match smooth_distributions:
            case 'convolution':
                hist_smoothed_nonzeros = convolution_smooth(hist_to_smooth_nonzeros, window_size)
            case 'gaussian':
                hist_smoothed_nonzeros = gaussian_smooth(hist_to_smooth_nonzeros, sigma)
            case _:
                raise ValueError(f"Unknown smoothing method: {smooth_distributions}")
        # add the zeros back
        hist_smoothed = hist_smoothed_nonzeros.copy()
        hist_smoothed[0] = zeros_bin
    return hist_bigger, hist_smoothed

# Calculate bin edges for datasets with different sizes
def freedman_diaconis(data, default_bins=10):
    if len(data) < 10:
        return default_bins
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    n = len(data)
    # Handling the case where IQR is 0, which could happen if all values are the same
    if iqr == 0:
        return default_bins
    bin_width = 2 * iqr * n ** (-1/3)
    data_range = np.max(data) - np.min(data)
    # Prevent division by zero or negative bin width
    if bin_width <= 0:
        return default_bins
    bin_count = data_range / bin_width
    # Ensure bin_count is at least 1 and rounded to the nearest whole number
    bin_count = max(1, np.round(bin_count))
    return int(bin_count)

def convolution_smooth(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return smoothed_data

def gaussian_smooth(data, sigma):
    # sigma is the standard deviation for the Gaussian kernel
    smoothed_values = gaussian_filter1d(data, sigma=sigma)
    return smoothed_values

def power_law_func(x, a, k, y0):
    return a*x**k + y0

def calculate_cell_number_dependent_jsd_decay(adata_sc, gene, celltype, initial_guess_pl):
    number_of_cells_to_sample = list(range(5, 1000, 20))
    number_of_samplings = 3

    mean_jsd = pd.DataFrame(columns=['cell_number', 'mean_jsd'])
    for cell_number in number_of_cells_to_sample:
        all_results = []
        for _ in range(number_of_samplings):
            # subset
            adata_sc_sample = adata_sc[adata_sc.obs['celltype'] == celltype].copy()
            sampled_indices = np.random.choice(adata_sc_sample.obs.index, cell_number, replace=True)
            adata_sc_sample = adata_sc_sample[adata_sc_sample.obs.index.isin(sampled_indices)]
            jsd_original = jensen_shannon_distance_per_gene_and_celltype(adata_sc=adata_sc, 
                                                                         adata_sp=adata_sc_sample,
                                                                         decay_param_df=None,
                                                                         gene=gene, 
                                                                         celltype=celltype, 
                                                                         smooth_distributions='no',
                                                                         correct_for_cell_number_dependent_decay=False)
            all_results.append({'cell_number': cell_number, 'mean_jsd': jsd_original})
        cell_number_vs_jsd = pd.DataFrame(all_results)
        mean_entry = cell_number_vs_jsd.mean(axis=0)
        mean_jsd = pd.concat([mean_jsd, pd.DataFrame([mean_entry])], ignore_index=True)
    
    try:
        # try to find a fit for the power law function
        popt_no_smoothing_pl, _ = curve_fit(power_law_func, mean_jsd['cell_number'],
                                                        mean_jsd['mean_jsd'], p0=initial_guess_pl, maxfev=2000)
    except Exception:
        return np.array([0, 1, 0]) # return a baseline that will not affect the JSD
    
    return popt_no_smoothing_pl

def generate_jsd_decay_data(adata_sc, initial_guess_pl, decay_csv_path):
    decay_params = pd.DataFrame(index=adata_sc.var_names, columns=adata_sc.obs['celltype'].unique())
    for gene in adata_sc.var_names:
        for celltype in adata_sc.obs['celltype'].unique():
            popt_no_smoothing_pl = calculate_cell_number_dependent_jsd_decay(adata_sc, gene, celltype, initial_guess_pl)
            decay_params.loc[gene, celltype] = popt_no_smoothing_pl
    decay_params.to_csv(decay_csv_path)
    return decay_params

def parse_array(array_str):
    array_str = array_str.strip('[]').split()
    numbers = [float(num) for num in array_str]
    return np.array(numbers)


# ONGOING
# TODO: maybe implement Wasserstein or maybe even the Cramer distance in addition to Jensen-Shannon
####