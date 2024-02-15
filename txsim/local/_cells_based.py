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
        bins: Tuple[int],
        cells_x_col: str = "x",
        cells_y_col: str = "y",)-> np.ndarray:
    """Get number of celltypes
    
    Parameters
    ---------
    adata_sp: AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    image : NDArray
        read from image of dapi stained cell-nuclei
    n_bins : List[int], optional
        The number of bins along the y and x axes, formatted as [ny, nx]. 
        Use either `bin_width` or `n_bins` to define grid cells.
        
    Returns
    -------
    A : np.ndarray of floats
        number of celltypes per bin
        one array per cell type, one array per x coordinate
        A[celltype][x coordinate][y coordinate]
    """
    
    #necessary to check if region is valid?
    # name of celltypes in adata all the same?
    
    celltypes = pd.Index(adata.obs["louvain"].unique())
    print(bins)
    A = np.zeros((len(celltypes),bins[0]+1,bins[1]+1)) # one dataframe for each celltype, with |x| lists of |y| variables. variables number of cell type(specific to
        #dataframe and bin
    
    for index, row in adata_sp.obs.iterrows():
        A[celltypes.get_loc(row['louvain']),row[cells_x_col],row[cells_y_col]] += 1
    
    return A
