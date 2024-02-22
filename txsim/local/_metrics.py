import numpy as np
import anndata as ad
from typing import Tuple
import pandas as pd

from ..metrics import knn_mixing_per_cell_score
from ..metrics import mean_proportion_deviation

#TODO: "negative_marker_purity_reads", "negative_marker_purity_cells", "coexpression_similarity", 
#    "relative_expression_similarity_across_genes", "relative_expression_similarity_across_celltypes",


def _get_knn_mixing_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    **kwargs
) -> np.ndarray:
    """Calculate the average knn mixing score within each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    adata_sc : AnnData
        Annotated AnnData object containing dissociated single cell transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
    kwargs : dict
        Additional keyword arguments for the txsim.metrics.knn_mixing_per_cell_score function.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the average knn mixing scores in each grid bin.
    """

    knn_mixing_per_cell_score(adata_sp, adata_sc, obs_key = obs_key, **kwargs)

    knn_mixing_score_key = kwargs["key_added"] if ("key_added" in kwargs) else "knn_mixing_score"
    
    df_cells = adata_sp.obs[[cells_y_col, cells_x_col, knn_mixing_score_key]]
    
    H = np.histogram2d(
        df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, 
        range=region_range, weights=df_cells[knn_mixing_score_key]
    )[0]
    
    # Normalize by the number of cells in each bin
    H = H / np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range)[0]
    
    return H


def _get_celltype_proportions_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    abs_score: bool = True,
    ct_set: str = "union",
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y"
) -> np.ndarray:
    """Calculate the average difference in cell type proportions within each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    adata_sc : AnnData
        Annotated AnnData object containing dissociated single cell transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    abs_score : bool, default True
        Whether to return the absolute score between 0 and 1 (higher means proportions more consistent with sc data)
        or a relative score between -1 and +1
        (negative score means cell types that are less abundant in spatial than sc data are more common per grid field,
        positive score cell types that are more abundant in spatial than sc data are more common).
    ct_set : str, default "union"
        How to combine the different cell types from both data sets.
        Supported: ["union", "intersection"]
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the average difference in cell type proportions in each grid bin.
    """

    # Get cell type proportion deviation scores per cell type
    _, df_props = mean_proportion_deviation(adata_sp, adata_sc, ct_set=ct_set, obs_key=obs_key, pipeline_output=False)

    # Map each cell's cell type to its corresponding proportion deviation
    df_cells = adata_sp.obs[[cells_x_col, cells_y_col, obs_key]]
    df_cells["sp_minus_sc"] = df_cells[obs_key].map(df_props["sp_minus_sc"]).astype(float)

    # Count number of cells per grid field for mean calculation
    ct_grid_counts = np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range)[0]

    # Sum cell scores per grid field
    ct_score_hist = np.full_like(ct_grid_counts, 0)
    ct_score_hist = np.histogram2d(
        df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range, 
        weights=df_cells["sp_minus_sc"].abs() if abs_score else df_cells["sp_minus_sc"]
    )[0]

    # Get mean scores by normalizing with number of cells per grid field
    ct_score_hist = ct_score_hist / ct_grid_counts

    return ct_score_hist
