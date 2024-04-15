import anndata as ad
from typing import Tuple, List
import numpy as np

def _get_spot_density_grid(
    adata_sp: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    spots_x_col: str = "x",
    spots_y_col: str = "y"
) -> np.ndarray:
    """Get the density of spots (RNA molecules) within each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    spots_x_col : str, default "x"
        The column name in adata_sp.obs["spots"] for the x-coordinates of spots.
    spots_y_col : str, default "y"
        The column name in adata_sp.obs["spots"] for the y-coordinates of spots.

    Returns
    -------
    H : np.ndarray
        A 2D numpy array representing the density of specified spot types in each grid bin.
    """
    assert "spots" in adata_sp.uns.keys(), "Spot annotation is missing in adata_sp.uns"

    df_spots = adata_sp.uns["spots"][[spots_y_col, spots_x_col]]
    H = np.histogram2d(df_spots[spots_y_col], df_spots[spots_x_col], bins=bins, range=region_range)[0]
    return H