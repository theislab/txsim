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


def get_summed_cell_area(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, bins: Tuple[int,int]):
    """Get summed cell area.
    
    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    Returns
    -------
    H : array of floats
        summed cell area
    range : range of binning 
    """
    df = adata_sp.obs
    range = (x_min,x_max,y_min,y_max)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    #filter spots
    df = df.loc[(df['x']>= x_min) & (df['x']<=x_max) & (df['y']>=y_min) & (df['y']<=y_max)]

    bins_x = np.digitize(df['x'], np.linspace(x_min, x_max, bins[0] + 1)) -1        
    bins_y = np.digitize(df['y'], np.linspace(y_min, y_max, bins[1] + 1)) -1

    groups = df.groupby([bins_x, bins_y])
    sums = groups['area'].sum()
    summed_cell_area = np.zeros((bins[0],bins[1]))

    for (i,j), value in sums.items():
        summed_cell_area[i,j] = value
    
    return summed_cell_area.T, range
