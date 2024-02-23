import numpy as np
import anndata as ad
from typing import Tuple

#TODO (fcts: _get_<...>_grid): 
# "celltype_density", "number_of_celltypes", "major_celltype_perc", "summed_cell_area", "spot_uniformity_within_cells"


def _get_cell_density_grid(
    adata_sp: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    cells_x_col: str = "x",
    cells_y_col: str = "y"
) -> np.ndarray:
    """Calculate the density of cells within each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the cell density in each grid bin.
    """
    
    df_cells = adata_sp.obs[[cells_y_col, cells_x_col]]
    H = np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range)[0]
    return H


def _get_number_of_celltypes(
        adata_sp: ad.AnnData,
        region_range: Tuple[Tuple[float, float], Tuple[float, float]],
        bins: Tuple[int, int],
        obs_key: str = "celltype",
        cells_x_col: str = "x",
        cells_y_col: str = "y"
):
    """Get number of celltypes

    Parameters
    ---------
    adata_sp: AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    n_bins : List[int], optional
        The number of bins along the y and x axes, formatted as [ny, nx].
        Use either `bin_width` or `n_bins` to define grid cells.
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
    Returns
    -------
    array2d :  np.ndarray
        A 2D numpy array representing the number of celltypes in each grid bin.
    """
    # calculate the cell density for every celltype
    density_per_celltype = _get_cell_density_grid_per_celltype(adata_sp, region_range, bins, obs_key, cells_x_col, cells_y_col)
    # stack density per celltype to a 3D array, 1st D: y bins, 2nd D: x bins, 3rd D: vector of length = number of celltypes
    histograms = list(density_per_celltype.values())
    histograms_3d = np.dstack(histograms)
    # check for every celltype if their density >0
    for i in range(len(histograms)):
        mask = histograms_3d[:, :, i] > 0
        histograms_3d[:, :, i][mask] = 1

    # sum up vectors of 3rd Dimension
    array2d = np.sum(histograms_3d, axis=2)
    return array2d
