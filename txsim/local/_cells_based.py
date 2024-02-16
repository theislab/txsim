import numpy as np
import anndata as ad
from typing import Tuple

from anndata import AnnData

from skimage.morphology import convex_hull_image

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


def _get_spot_uniformity_within_cells_grid(
    adata_sp: AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    spots_x_col: str = "x",
    spots_y_col: str = "y",
    **kwargs
) -> np.ndarray:
    """Calculate the average spot uniformity within cells for each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData containing spatial transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
    spots_x_col : str, default "x"
        The column name in adata_sp.uns["spots"] for the x-coordinates of spots.
    spots_y_col : str, default "y"
        The column name in adata_sp.uns["spots"] for the y-coordinates of spots.
    kwargs : dict, optional
        Additional keyword arguments for the spot_uniformity_per_cell_score function.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the average spot uniformity scores in each grid bin.
    """

    spot_uniformity_per_cell_score(adata_sp, spots_x_col=spots_x_col, spots_y_col=spots_y_col, **kwargs)

    uniform_cell_key = kwargs["key_added"] if ("key_added" in kwargs) else "uniform_cell"

    df_cells = adata_sp.obs[[cells_y_col, cells_x_col, uniform_cell_key]]

    H = np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col],
                       bins=bins, range=region_range, weights=df_cells[uniform_cell_key])[0]

    # Normalize by the number of cells in each bin
    H = H / np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range)[0]

    return H


def spot_uniformity_per_cell_score(
    adata_sp: AnnData,
    spots_x_col: str = "x",
    spots_y_col: str = "y",
    key_added: str = "uniform_cell"
) -> None:
    """Compute how uniform spots are distributed over cells.

    Therefore, we compare the observed and expected counts using the chi-quadrat-statistic.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data.
    spots_x_col : str, default "x"
        The column name in the spots table for the x-coordinates of spots.
    spots_y_col : str, default "y"
        The column name in the spots table for the y-coordinates of spots.
    key_added : str, default "uniform_cell"
        adata_sp.obs key where uniformity scores are saved.

    Returns
    -------
    nothing - just added scores to `adata.obs[key_added]`
    """

    df = adata_sp.uns["spots"]
    unique_cell_ids = adata_sp.obs.index.unique()
    adata_sp.obs[key_added] = np.nan

    for i in unique_cell_ids:
        spots = df.loc[df["cell_id"] == i].copy()
        spots[spots_x_col], spots[spots_y_col] = [spots[spots_x_col].round().astype(int),
                                                  spots[spots_y_col].round().astype(int)]

        if len(spots[spots_x_col]) or len(spots[spots_y_col]):
            [x_min, x_max, y_min, y_max] = [np.nanmin(spots[spots_x_col]), np.nanmax(spots[spots_x_col]),
                                            np.nanmin(spots[spots_y_col]), np.nanmax(spots[spots_y_col])]
            spots[spots_x_col], spots[spots_y_col] = spots[spots_x_col] - x_min, spots[spots_y_col] - y_min

            seg_mask = np.zeros((x_max - x_min + 1, y_max - y_min + 1))
            seg_mask[spots[spots_x_col].values.tolist(), spots[spots_y_col].values.tolist()] = 1
            cell = convex_hull_image(seg_mask)

            # Define the number of quadrats in each dimension
            n_quadrats_x, n_quadrats_y = x_max - x_min + 1, y_max - y_min + 1
            # Count the number of spots in each quadrat
            quadrat_counts = np.histogram2d(spots[spots_x_col], spots[spots_y_col],
                                            bins=[n_quadrats_x, n_quadrats_y])[0]

            # observed and expected counts
            observed_counts = quadrat_counts[cell]
            total_spots = len(spots)
            n_pixs = np.sum(cell)
            mean_pix = total_spots / n_pixs
            expected_counts = np.full_like(observed_counts, mean_pix)

            # Calculate the Chi-squared statistic
            chi2_statistic = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)

            # delta peak: all spots in one pixel
            chi2_delta = (n_pixs - 1) * mean_pix + (total_spots - mean_pix) ** 2 / mean_pix

            # Calculate a uniformness measure based on the Chi-squared statistic
            adata_sp.obs.loc[adata_sp.obs.index == i, key_added] = 1 if chi2_delta == 0 \
                else 1 - chi2_statistic / chi2_delta
