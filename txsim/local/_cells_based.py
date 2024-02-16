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


def get_summed_cell_area(adata_sp: AnnData, region_range: Tuple[Tuple[float, float], Tuple[float, float]], bins: Tuple[int,int]):
    """Get summed cell area.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    region_range: Tuple[Tuple[float,float],Tuple[float,float]]
        Crop coordinates
    bins : Tuple[int,int]
        The number of bins in each dimension
        
    Returns
    -------
    summed_cell_area : array
        Summed cell area
    """
    df = adata_sp.obs
    x_min = region_range[0][0]
    x_max = region_range[0][1]
    y_min = region_range[1][0]
    y_max = region_range[1][1]
    range = (x_min, x_max, y_min, y_max)

    # Filter spots within region range
    df = df.loc[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]

    # Calculate histogram
    summed_cell_area = np.histogram2d(df['x'], df['y'], bins=bins, range=[[x_min, x_max], [y_min, y_max]])[0]

    return summed_cell_area.T
