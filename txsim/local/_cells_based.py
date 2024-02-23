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


def get_summed_cell_area(adata_sp: ad, region_range: Tuple[Tuple[float, float], Tuple[float, float]], bins: Tuple[int,int], cells_x_col: str = "x", cells_y_col: str = "y"):
    """Get summed cell area.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    region_range: Tuple[Tuple[float,float],Tuple[float,float]]
        Crop coordinates
    bins : Tuple[int,int]
        The number of bins in each dimension
    str x : cells_x_col:
        Colums of x values
    sry y : cells_y_col:
        Colums of y values

    Returns
    -------
    summed_cell_area : array
        Summed cell area
    """
    df = adata_sp.obs
    
    H, x_edges, y_edges = np.histogram2d(df[cells_x_col], df[cells_y_col], bins=bins, range=region_range)

    bin_indices_x = np.digitize(df['x'], x_edges) - 1
    bin_indices_y = np.digitize(df['y'], y_edges) - 1


    summed_cell_area = np.zeros((len(x_edges) , len(y_edges) ))

    np.add.at(summed_cell_area, (bin_indices_x, bin_indices_y), df['area'])

    return summed_cell_area.T
