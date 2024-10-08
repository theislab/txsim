import numpy as np
import pandas as pd
import anndata as ad
from typing import List, Dict, Tuple, Optional, Union

from ._cells_based import _get_cell_density_grid, _get_number_of_celltypes_grid, _get_cell_density_grid_per_celltype
from ._cells_based import _get_celltype_ratio_grid, _get_spot_uniformity_within_cells_grid, _get_summed_cell_area_grid
from ._spots_based import _get_spot_density_grid
from ._metrics import _get_knn_mixing_grid, _get_celltype_proportions_grid
from ._metrics import _get_relative_expression_similarity_across_genes_grid
from ._metrics import _get_relative_expression_similarity_across_celltypes_grid
from ._self_consistency_metrics import _get_ARI_between_cell_assignments_grid
from ._self_consistency_metrics import _get_spots_based_annotation_similarity_grid

SUPPORTED_CELL_AND_SPOT_STATISTICS = [
    "cell_density", "spot_density", "cell_density_per_celltype", "celltype_percentage", "number_of_celltypes", 
    "major_celltype_perc", "summed_cell_area", "spot_uniformity_within_cells"
]
SUPPORTED_IMAGE_FEATURES = []
SUPPORTED_QUALITY_METRICS = []
SUPPORTED_METRICS = [
    "negative_marker_purity_reads", "negative_marker_purity_cells", "knn_mixing", "coexpression_similarity", 
    "relative_expression_similarity_across_genes", "relative_expression_similarity_across_celltypes",
    "celltype_proportions"
]
SUPPORTED_SELF_CONSISTENCY_METRICS = [
    "ARI_cell_assignments", "annotation_similarity"
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
        eps = 1e-6
        y_min, y_max = spots[spots_y_col].min(), spots[spots_y_col].max() + eps
        x_min, x_max = spots[spots_x_col].min(), spots[spots_x_col].max() + eps
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
        Supported metrics include: ["cell_density", "spot_density", "celltype_density", "number_of_celltypes",
                                    "major_celltype_perc", "summed_cell_area", "spot_uniformity_within_cells"].
    grid_region : List[Union[float, List[float]]], optional
        The spatial domain over which to set the grid. Options include:
        1. [y_max, x_max] (e.g., the shape of the associated DAPI image).
        2. [[y_min, y_max], [x_min, x_max]] (e.g., coordinates of a cropped area -> grid: xy_min <= xy < xy_max).
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
        
    if "summed_cell_area" in metrics:
        out_dict["summed_cell_area"] = _get_summed_cell_area_grid(
            adata_sp, region_range, bins, area_key="area", cells_x_col=cells_x_col, cells_y_col=cells_y_col
        )
        
    if "number_of_celltypes" in metrics:
        out_dict["number_of_celltypes"] = _get_number_of_celltypes_grid(
            adata_sp, region_range, bins, obs_key, cells_x_col, cells_y_col
        )
        
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
            
    if "spot_density" in metrics:
        out_dict["spot_density"] = _get_spot_density_grid(adata_sp, region_range, bins, spots_x_col, spots_y_col)
        
    if "spot_uniformity_within_cells" in metrics:
        out_dict["spot_uniformity_within_cells"] = _get_spot_uniformity_within_cells_grid(
            adata_sp, region_range, bins, cells_x_col, cells_y_col, spots_x_col, spots_y_col
        )
           
    return out_dict, grid_coords

def image_features(
    image: np.ndarray,
    adata_sp: Optional[ad.AnnData] = None,
    metrics: Union[str, List[str]] = "all",
    grid_region: Optional[List[Union[float, List[float]]]] = None,
    bin_width: Optional[float] = None,
    n_bins: Optional[List[int]] = None,
    spots_x_col: str = "x",
    spots_y_col: str = "y"
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute image features over a spatial grid.

    Parameters
    ----------
    image : NDArray
        read from image of dapi stained cell-nuclei
    adata_sp : AnnData, optional
        Annotated AnnData object containing spatial transcriptomics data. 
        The spots' coordinates should be in adata.uns["spots"][[spots_x_col, spots_y_col]]
    metrics : str or List[str], default "all"
        The metrics to compute. Specify "all" to compute all available metrics or provide a list of specific metrics. 
        Supported metrics include: [TODO]. #NOTE: Add supported metrics
    grid_region : List[Union[float, List[float]]], optional
        The spatial domain over which to set the grid. Options include:
        1. [y_max, x_max] (e.g., the shape of the associated DAPI image).
        2. [[y_min, y_max], [x_min, x_max]] (e.g., coordinates of a cropped area -> grid: xy_min <= xy < xy_max).
        3. None (if None and adata_sp given, the grid is inferred from the min and max spots' coordinates, if adata_sp not given, the grid is inferred from the shape of image).
    bin_width : float, optional
        The width of each grid field. Use either `bin_width` or `n_bins` to define grid cells.
    n_bins : List[int], optional
        The number of bins along the y and x axes, formatted as [ny, nx]. 
        Use either `bin_width` or `n_bins` to define grid cells.
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
    # set grid_region
    if (grid_region is None) and (adata_sp is None):
        grid_region = list(image.shape)
        
    # Set metrics
    metrics = _convert_metrics_input_to_list(metrics, SUPPORTED_IMAGE_FEATURES)

    # Set grid region
    spots = adata_sp.uns["spots"] if "spots" in adata_sp.uns else None # Some metrics can be run without spots
    region_range, bins = _convert_grid_specification_to_range_and_bins(
        spots, grid_region, bin_width, n_bins, spots_x_col, spots_y_col
    )
    grid_coords = _convert_range_and_bins_to_grid_coordinates(region_range, bins)
    
    # Compute metrics
    out_dict = {}
    # if "metric1" in metrics:
    #    out_dict["metric2"] = _get_metric_1(image, region_range, bins)
           
    return out_dict, grid_coords


def self_consistency_metrics(
        adata_sp1: ad.AnnData,
        adata_sp2: ad.AnnData,
        metrics: Union[str, List[str]] = "all",
        grid_region: Optional[List[Union[float, List[float]]]] = None,
        bin_width: Optional[float] = None,
        n_bins: Optional[List[int]] = None,
        obs_key: str = "celltype",
        uns_key: str = "spots",
        ann_key: str = "cell_id",
        spots_ct_key: str = "celltype",
        cells_x_col: str = "x",
        cells_y_col: str = "y",
        spots_x_col: str = "x",
        spots_y_col: str = "y",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute similarity statistics for two spatial datasets over a spatial grid.

    This function calculates various spatial statistics for comparing two spatial transcriptomic datasets across a defined grid.
    It allows customization of the grid and selection of specific metrics for detailed spatial analysis.

    Parameters
    ----------
    adata_sp1 : AnnData
        First annotated AnnData object containing spatial transcriptomics data.
        The spots' coordinates should be in adata_sp1.uns["spots"][[spots_x_col, spots_y_col]],
        and the cells' coordinates in adata_sp1.obs[[cells_x_col, cells_y_col]].
    adata_sp2 : AnnData
        Second annotated AnnData object containing spatial transcriptomics data.
        Requirements are the same as for adata_sp1.
    metrics : str or List[str], default "all"
        The metrics to compute. Specify "all" to compute all available metrics or provide a list of specific metrics.
        Supported metrics include: ["ARI_spot_clusters", "annotation_similarity"].
    grid_region : List[Union[float, List[float]]], optional
        The spatial domain over which to set the grid. The same grid is used for both spatial datasets. Options include:
        1. [y_max, x_max] (e.g., the shape of the associated DAPI image).
        2. [[y_min, y_max], [x_min, x_max]] (e.g., coordinates of a cropped area -> grid: xy_min <= xy < xy_max).
        3. None (if None, the grid is inferred from the min and max spots' coordinates).
    bin_width : float, optional
        The width of each grid field. Use either `bin_width` or `n_bins` to define grid cells.
    n_bins : List[int], optional    
        The number of bins along the y and x axes, formatted as [ny, nx].
        Use either `bin_width` or `n_bins` to define grid cells.
    obs_key : str, default "celltype"
        The column name in adata.obs for the cell type annotations. Must be the same for both datasets.
    uns_key : str
        Key where to find the data containing the spots information in both adata.uns
    ann_key : str
        Key where the annotation for teh cell IDs are found in adata.uns[uns_key]
    spots_ct_key : str
        The column name in adata.uns[uns_key] for the cell type annotations. Must be the same for both datasets.
    cells_x_col : str, default "x"
        The column name in adata.obs for the x-coordinates of cells. Must be the same for both datasets.
    cells_y_col : str, default "y"
        The column name in adata.obs for the y-coordinates of cells. Must be the same for both datasets.
    spots_x_col : str, default "x"
        The column name in adata.uns[uns_key] for the x-coordinates of spots. Must be the same for both datasets.
    spots_y_col : str, default "y"
        The column name in adata.uns[uns_key] for the y-coordinates of spots. Must be the same for both datasets.

    Returns
    -------
    Dict[str, np.ndarray]
        A tuple containing the calculated statistics. The first element is a dictionary with each metric's name as keys
        and their corresponding numpy arrays as values.
    np.ndarray
        The second element is a numpy array representing the coordinates of the grid used for calculations.

    """

    # Set metrics
    metrics = _convert_metrics_input_to_list(metrics, SUPPORTED_SELF_CONSISTENCY_METRICS)

    # Set grid region using the first spatial dataset, but the same grid is used for both datasets
    spots = adata_sp1.uns["spots"] if "spots" in adata_sp1.uns else None  # Some metrics can be run without spots
    region_range, bins = _convert_grid_specification_to_range_and_bins(
        spots, grid_region, bin_width, n_bins, spots_x_col, spots_y_col
    )
    grid_coords = _convert_range_and_bins_to_grid_coordinates(region_range, bins)
    
    # Compute metrics
    out_dict = {}
    if "ARI_cell_assignments" in metrics:
        out_dict["ARI_cell_assignments"] = _get_ARI_between_cell_assignments_grid(
            adata_sp1, adata_sp2, region_range, bins, uns_key, ann_key, spots_x_col, spots_y_col
        )
    if "annotation_similarity" in metrics:
        out_dict["annotation_similarity"] = _get_spots_based_annotation_similarity_grid(
            adata_sp1, adata_sp2, region_range, bins, uns_key, spots_ct_key, spots_x_col, spots_y_col
        )
    
    return out_dict, grid_coords


def metrics(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    metrics: Union[str, List[str]] = "all",
    grid_region: Optional[List[Union[float, List[float]]]] = None,
    bin_width: Optional[float] = None,
    n_bins: Optional[List[int]] = None,
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    spots_x_col: str = "x",
    spots_y_col: str = "y",
    layer: str = 'lognorm',
    normalization: str = "global",
    contribution: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute similarity metrics between spatial and dissociated data over a spatial grid.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data. 
        The spots' coordinates should be in adata.uns["spots"][[spots_x_col, spots_y_col]], 
        and the cells' coordinates in adata.obs[[cells_x_col, cells_y_col]]. 
    adata_sc : AnnData
        Annotated AnnData object containing dissociated single-cell transcriptomics data.
    metrics : str or List[str], default "all"
        The metrics to compute. Specify "all" to compute all available metrics or provide a list of specific metrics. 
        Supported metrics include: [TODO]. #NOTE: Add supported metrics
    grid_region : List[Union[float, List[float]]], optional
        The spatial domain over which to set the grid. Options include:
        1. [y_max, x_max] (e.g., the shape of the associated DAPI image).
        2. [[y_min, y_max], [x_min, x_max]] (e.g., coordinates of a cropped area -> grid: xy_min <= xy < xy_max).
        3. None (if None, the grid is inferred from the min and max spots' coordinates).
    bin_width : float, optional
        The width of each grid field. Use either `bin_width` or `n_bins` to define grid cells.
    n_bins : List[int], optional
        The number of bins along the y and x axes, formatted as [ny, nx]. 
        Use either `bin_width` or `n_bins` to define grid cells.
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata.obs for the y-coordinates of cells.
    spots_x_col : str, default "x"
        The column name in adata.uns["spots"] for the x-coordinates of spots.
    spots_y_col : str, default "y"
        The column name in adata.uns["spots"] for the y-coordinates of spots.
    layer: str (default: 'lognorm')
        Applicable only for the relative_expression_similarity metrics.
        Layer of ``AnnData`` to use to compute the metric.
    normalization: str (default: "global")
        Applicable only for the relative_expression_similarity metrics.
        The type of normalization to use for computing the metric. If set to "global"'", the entire spatial dataset is used
        for normalization; if set to "local", only the local grid field is used to calculate the normalization factor.
        Can be either "global" or "local".
    contribution: bool (default: True)
        Applicable only for the relative_expression_similarity metrics.
        Set to True to calculate the contribution of each grid field to the overall metric, or False to calculate the metric itself.

    Returns
    -------
    Dict[str, np.ndarray] 
        A tuple containing the calculated metrics. The first element is a dictionary with each metric's name as keys 
        (note that some metrics might be converted to multiple keys, e.g. celltype_density -> celltype_density_Tcells, #TODO: check for example in metrics, otherwise remove
        celltype_density_Bcells, ...) and their corresponding numpy arrays as values. 
    np.ndarray
        Numpy array representing the coordinates of the grid used for calculations.

    """
    
    # Set metrics
    metrics = _convert_metrics_input_to_list(metrics, SUPPORTED_METRICS)
    
    # Set grid region
    spots = adata_sp.uns["spots"] if "spots" in adata_sp.uns else None # Some metrics can be run without spots
    region_range, bins = _convert_grid_specification_to_range_and_bins(
        spots, grid_region, bin_width, n_bins, spots_x_col, spots_y_col
    )
    grid_coords = _convert_range_and_bins_to_grid_coordinates(region_range, bins)
    
    # Compute metrics
    out_dict = {}
    if "negative_marker_purity_reads" in metrics:
        raise NotImplementedError("negative_marker_purity_reads is not yet implemented.")
        #out_dict["negative_marker_purity_reads"] = _get_negative_marker_purity_reads_grid(
        #    adata_sp, adata_sc, ct_key, region_range, bins, cells_x_col, cells_y_col
        #)
    if "negative_marker_purity_cells" in metrics:
        raise NotImplementedError("negative_marker_purity_cells is not yet implemented.")
        #out_dict["negative_marker_purity_cells"] = _get_negative_marker_purity_cells_grid(
        #    adata_sp, adata_sc, ct_key, region_range, bins, cells_x_col, cells_y_col
        #)
    if "knn_mixing" in metrics:
        out_dict["knn_mixing"] = _get_knn_mixing_grid(
            adata_sp.copy(), adata_sc.copy(), region_range, bins, obs_key, cells_x_col, cells_y_col)
    if "coexpression_similarity" in metrics:
        raise NotImplementedError("coexpression_similarity is not yet implemented.")
        #out_dict["coexpression_similarity"] = _get_coexpression_similarity_grid(adata_sp, adata_sc, ct_key, region_range, bins, cells_x_col, cells_y_col)
    if "relative_expression_similarity_across_genes" in metrics:
        out_dict["relative_expression_similarity_across_genes"] = _get_relative_expression_similarity_across_genes_grid(
            adata_sp, adata_sc, region_range, bins, obs_key, layer, cells_x_col, cells_y_col, normalization,
            contribution
        )
    if "relative_expression_similarity_across_celltypes" in metrics:
        out_dict["relative_expression_similarity_across_celltypes"] = _get_relative_expression_similarity_across_celltypes_grid(
            adata_sp, adata_sc, region_range, bins, obs_key, layer, cells_x_col, cells_y_col, normalization, 
            contribution
        )
    if "celltype_proportions" in metrics:
        out_dict["celltype_proportions_abs_diff"] = _get_celltype_proportions_grid(
            adata_sp.copy(), adata_sc.copy(), region_range, bins, abs_score=True,
            obs_key=obs_key, cells_x_col=cells_x_col, cells_y_col=cells_y_col)
        out_dict["celltype_proportions_diff"] = _get_celltype_proportions_grid(
            adata_sp.copy(), adata_sc.copy(), region_range, bins, abs_score=False,
            obs_key=obs_key, cells_x_col=cells_x_col, cells_y_col=cells_y_col)
           
    return out_dict, grid_coords


#TODO: Implement the following wrapper functions    
# - tx.local.quality_metrics(adata_sp)
