import numpy as np
import anndata as ad
from typing import Tuple
import pandas as pd
from scipy.sparse import issparse

from ..metrics import jensen_shannon_distance
from ._utils import _get_bin_ids

def _get_jensen_shannon_distance_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    obs_key: str = "celltype",
    layer: str = 'lognorm',
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    min_number_cells:int=10, # the minimal number of cells per celltype to be considered
    smooth_distributions:str='no_smoothing',
    window_size:int=7,
    sigma:int=2,
    correct_for_cell_number_dependent_decay:bool=False,
    filter_out_double_zero_distributions:bool=True,
    decay_csv_enclosing_folder='output'
):
### SET UP
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer] = a.layers[layer].toarray()

    # get bin ids
    adata_sp.obs = _get_bin_ids(adata_sp.obs, region_range, bins, cells_x_col, cells_y_col)

    # only consider cells within the specified region
    adata_sp_region_range = adata_sp[(adata_sp.obs["y_bin"] != -1) & (adata_sp.obs["x_bin"] != -1)]

    # create an empty matrix to store the computed metric for each grid field
    overall_metric_matrix = np.zeros((bins[0], bins[1]))

    for y_bin in adata_sp_region_range.obs["y_bin"].unique():
        for x_bin in adata_sp_region_range.obs["x_bin"].unique():
            # subset the spatial data to only include cells in the current grid field
            adata_sp_local = adata_sp_region_range[(adata_sp_region_range.obs["y_bin"] == y_bin) & 
                                                   (adata_sp_region_range.obs["x_bin"] == x_bin)]

            # pipeline output=True, # TODO add functions per gene and per celltype
            jsd = jensen_shannon_distance(adata_sc = adata_sc, 
                                    adata_sp = adata_sp_local,
                                    key=obs_key, 
                                    layer=layer, 
                                    min_number_cells=min_number_cells,
                                    smooth_distributions=smooth_distributions,
                                    window_size=window_size,
                                    sigma=sigma, 
                                    pipeline_output=True,
                                    correct_for_cell_number_dependent_decay=correct_for_cell_number_dependent_decay,
                                    filter_out_double_zero_distributions=filter_out_double_zero_distributions,
                                    decay_csv_enclosing_folder=decay_csv_enclosing_folder)
            overall_metric_matrix[y_bin, x_bin]  = jsd

    return overall_metric_matrix