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


def major_celltype_perc(
    adata_sp: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
)-> np.ndarray:
    """calculates most common celltype (percentage) for each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    Returns
    -------
    np.ndarray
        A 2D numpy array representing the percentage of the most common  cell type in each grid bin.
    """
    celltypes = adata_sp.obs["celltype"].unique()
    grid_major_celltype_empty = np.zeros((len(celltypes),bins[0],bins[1]))
    
    for i in range(len(celltypes)):
        grid_major_celltype_empty[i,...] = get_celltype_density(adata_sp, celltypes[i],region_range[1][0],region_range[1][1], region_range[0][0],region_range[0][1],bins)[0]
    grid_major_celltype_perc = np.max(grid_major_celltype_empty,axis=0)

    return grid_major_celltype_perc
