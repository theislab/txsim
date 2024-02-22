import numpy as np
import anndata as ad
from typing import Tuple, Dict

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

def _get_cell_density_grid_per_celltype(
    adata_sp: ad.AnnData, 
    region_range: Tuple[Tuple[float, float], Tuple[float, float]], 
    bins: Tuple[int, int], 
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y"
) -> Dict[str, np.ndarray]:
    """Calculate the density of cells within each grid bin for each cell type.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
        
    Returns
    -------
    dict of np.ndarrays
        key: celltype 
        value: 2D numpy array representing the cell density of the given cell type in each grid bin.

    """
    df =  adata_sp.obs

    celltypes = df[obs_key].unique()

    H_dict = {}
    for celltype in celltypes:
      df_filtered = df.loc[df[obs_key]==celltype]
      H_celltype = np.histogram2d(df_filtered[cells_y_col], df_filtered[cells_x_col], bins=bins, range=region_range)[0]
      H_dict[celltype] = H_celltype

    return H_dict


def _get_celltype_ratio_grid(
    adata_sp: ad.AnnData, 
    region_range: Tuple[Tuple[float, float], Tuple[float, float]], 
    bins: Tuple[int, int], 
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y"
) -> Dict[str, np.ndarray]:
    """Calculate the cell type ratio within each grid bin for each cell type.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
        
    Returns
    -------
    dict of np.ndarrays
        key: celltype 
        value: 2D numpy array representing the cell type ratio of the given cell type in each grid bin.

    """
    df =  adata_sp.obs

    celltypes = df[obs_key].unique()

    H_total = np.histogram2d(df[cells_y_col],df[cells_x_col], bins=bins, range=region_range)[0]

    H_dict = {}
    for celltype in celltypes:
      df_filtered = df.loc[df[obs_key]==celltype]
      H_celltype = np.histogram2d(df_filtered[cells_y_col], df_filtered[cells_x_col], bins=bins, range=region_range)[0]
      H = H_celltype/H_total
      H[np.isnan(H)] = 0
      H_dict[celltype] = H

    return H_dict
