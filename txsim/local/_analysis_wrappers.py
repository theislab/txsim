import numpy as np
import pandas as pd
import anndata as ad
from typing import List, Dict, Tuple, Optional, Union

from ._cells_based import _get_cell_density_grid, _get_cell_density_grid_per_celltype, _get_celltype_ratio_grid


SUPPORTED_CELL_AND_SPOT_STATISTICS = [
    "cell_density", "spot_density", "cell_density_per_celltype", "celltype_percentage", "number_of_celltypes", 
    "major_celltype_perc", "summed_cell_area", "spot_uniformity_within_cells"
]
SUPPORTED_IMAGE_FEATURES = []
SUPPORTED_QUALITY_METRICS = []
SUPPORTED_METRICS = [
    "negative_marker_purity_reads", "negative_marker_purity_cells", "knn_mixing", "coexpression_similarity", 
    "relative_expression_similarity_across_genes", "relative_expression_similarity_across_celltypes",
]
SUPPORTED_SELF_CONSISTENCY_METRICS = [
    "ARI_spot_clusters", "annotation_similarity" 
]



def _convert_metrics_input_to_list(metrics: Union[str, List[str]], supported: List[str]) -> List[str]:
    """Helper function: Convert metrics input to list."""
    
    if metrics == "all":
        metrics = supported
    elif isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)
        
    unsupported_metrics = [m for m in metrics if m not in supported]
    assert len(unsupported_metrics) == 0, f"Unsupported metrics: {unsupported_metrics}"
    
    return metrics

def _convert_grid_specification_to_range_and_bins(
        spots: pd.DataFrame,
        grid_region: Optional[List[Union[float, List[float]]]],
        bin_width: Optional[float],
        n_bins: Optional[List[int]],
        spots_x_col: str = "x",
        spots_y_col: str = "y",
    ) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[int, int]]:
    """Helper function: Convert grid specification to range and bins."""
    
    assert (bin_width is None) != (n_bins is None), "Either bin_width or n_bins must be provided."
    
    if grid_region is None:
        y_min, y_max = spots[spots_y_col].min(), spots[spots_y_col].max()
        x_min, x_max = spots[spots_x_col].min(), spots[spots_x_col].max()
    elif isinstance(grid_region, list) and (len(grid_region) == 2):
        if isinstance(grid_region[0], list) and isinstance(grid_region[1], list):
            y_min, y_max = grid_region[0]
            x_min, x_max = grid_region[1]
        else:
            y_min = x_min = 0
            y_max = grid_region[0]
            x_max = grid_region[1]
            
    if bin_width is not None:
        bins = (int((y_max+1 - y_min) / bin_width), int((x_max+1 - x_min) / bin_width))
    else:
        bins = n_bins
        
    return ((y_min, y_max), (x_min, x_max)), bins
        
def _convert_range_and_bins_to_grid_coordinates(
        range: Tuple[Tuple[float, float], Tuple[float, float]], 
        bins: Tuple[int, int]
    ) -> np.ndarray:
    """Helper function: Convert range and bins to grid coordinates."""
    
    return np.meshgrid(np.linspace(*range[1], bins[1]), np.linspace(*range[0], bins[0]))


#####################
# Wrapper functions #
#####################

def cell_and_spot_statistics(
    adata_sp: ad.AnnData,
    metrics: Union[str, List[str]] = "all",
    grid_region: Optional[List[Union[float, List[float]]]] = None,
    bin_width: Optional[float] = None,
    n_bins: Optional[List[int]] = None,
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    spots_x_col: str = "x",
    spots_y_col: str = "y",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute cell and spot statistics over a spatial grid.

    This function calculates various spatial statistics for cells and spots across a defined grid. 
    It allows customization of the grid and selection of specific metrics for detailed spatial analysis.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data. 
        The spots' coordinates should be in adata.uns["spots"][[spots_x_col, spots_y_col]], 
        and the cells' coordinates in adata.obs[[cells_x_col, cells_y_col]]. 
    metrics : str or List[str], default "all"
        The metrics to compute. Specify "all" to compute all available metrics or provide a list of specific metrics. 
        Supported metrics include: [TODO]. #NOTE: Add supported metrics
    grid_region : List[Union[float, List[float]]], optional
        The spatial domain over which to set the grid. Options include:
        1. [y_max, x_max] (e.g., the shape of the associated DAPI image).
        2. [[y_min, y_max], [x_min, x_max]] (e.g., coordinates of a cropped area -> grid: xy_min <= xy <= xy_max).
        3. None (if None, the grid is inferred from the min and max spots' coordinates).
    bin_width : float, optional
        The width of each grid field. Use either `bin_width` or `n_bins` to define grid cells.
    n_bins : List[int], optional
        The number of bins along the y and x axes, formatted as [ny, nx]. 
        Use either `bin_width` or `n_bins` to define grid cells.
    obs_key : str, default "celltype"
        The column name in adata_sp.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata.obs for the y-coordinates of cells.
    spots_x_col : str, default "x"
        The column name in adata.uns["spots"] for the x-coordinates of spots.
    spots_y_col : str, default "y"
        The column name in adata.uns["spots"] for the y-coordinates of spots.

    Returns
    -------
    Dict[str, np.ndarray] 
        A tuple containing the calculated statistics. The first element is a dictionary with each metric's name as keys 
        (note that some metrics might be converted to multiple keys, e.g. celltype_density -> celltype_density_Tcells,
        celltype_density_Bcells, ...) and their corresponding numpy arrays as values. 
    np.ndarray
        The second element is a numpy array representing the coordinates of the grid used for calculations.

    """
    
    # Set metrics
    metrics = _convert_metrics_input_to_list(metrics, SUPPORTED_CELL_AND_SPOT_STATISTICS)
    
    # Set grid region
    spots = adata_sp.uns["spots"] if "spots" in adata_sp.uns else None # Some metrics can be run without spots
    region_range, bins = _convert_grid_specification_to_range_and_bins(
        spots, grid_region, bin_width, n_bins, spots_x_col, spots_y_col
    )
    grid_coords = _convert_range_and_bins_to_grid_coordinates(region_range, bins)
    
    # Compute metrics
    out_dict = {}
    if "cell_density" in metrics:
        out_dict["cell_density"] = _get_cell_density_grid(adata_sp, region_range, bins, cells_x_col, cells_y_col)
    if "cell_density_per_celltype" in metrics:
        density_grid_dict = _get_cell_density_grid_per_celltype(
            adata_sp, region_range, bins, obs_key, cells_x_col, cells_y_col
        )
        for ct, density_grid in density_grid_dict.items():
            out_dict[f"cell_density_{ct}"] = density_grid
    if "celltype_percentage" in metrics:
        density_grid_dict = _get_celltype_ratio_grid(
            adata_sp, region_range, bins, obs_key, cells_x_col, cells_y_col
        )
        for ct, density_grid in density_grid_dict.items():
            out_dict[f"celltype_percentage_{ct}"] = density_grid
            
           
    return out_dict, grid_coords
        



#TODO: Implement the following wrapper functions    
# - tx.local.image_features(image, adata_sp=None) # adata_sp needs to be given if grid_region is None
# - tx.local.quality_metrics(adata_sp)
# - tx.local.metrics(adata_sp, adata_sc)
# - tx.local.self_consistency_metrics(adata_sp1, adata_sp2)