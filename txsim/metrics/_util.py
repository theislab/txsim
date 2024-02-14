from anndata import AnnData
import numpy as np
# from ._negative_marker_purity import get_spot_assignment_col
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse import isspmatrix 
import warnings
from typing import Tuple

#helper function 
import numpy as np

def check_crop_exists(image: np.ndarray | Tuple[int, int],
                      x_min: int = None, 
                      x_max: int = None, 
                      y_min: int = None,
                      y_max: int = None):
    """Check if crop coordinates are valid or use the entire image.
    
    This function checks if valid crop coordinates are provided. If not, and an image is provided,
    it defaults to using the entire image dimensions as the crop coordinates.
    
    Parameters
    ----------
    x_min, x_max, y_min, y_max : int
        Crop coordinates.
    image : np.ndarray | Tuple[int, int]
        Image from which the crop is to be taken or the size of the image as a tuple of integers.
        The coordinates are [image.shape[0],image.shape[1]] or [height, width].
    
    Returns
    -------
    crop_coordinates : list of lists
        The validated or computed crop coordinates [[x_min, x_max], [y_min, y_max]].
    
    Raises
    ------
    ValueError
        If crop coordinates are invalid or neither coordinates nor an image is provided.
    """
    if isinstance(image, np.ndarray):
        # Process the image
        image_x, image_y = image.shape[1], image.shape[0]
    elif isinstance(image, tuple) and len(image) == 2 and all(isinstance(i, int) for i in image):
        # Process the size
        image_x, image_y = image[1], image[0]
    else:
        raise ValueError("Invalid image parameter format. Provide an image or its size as a tuple of integers.")

    # Validate provided coordinates
    if any(coord is None for coord in [x_min, x_max, y_min, y_max]):
        # Use full image if coordinates are not provided
        x_min, x_max, y_min, y_max = 0, image_x, 0, image_y
        warnings.warn("No crop coordinates provided. Using the entire image dimensions as crop coordinates.")
    else:
        # Validate the logic of the coordinates
        if x_max <= x_min or y_max <= y_min:
            print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
            raise ValueError("x_max must be larger than x_min and y_max must be larger than y_min.")
        if x_min < 0 or y_min < 0:
            raise ValueError("x_min and y_min must be equal to or greater than 0.")
        if x_max > image_x or y_max > image_y:
            raise ValueError("x_max and y_max must be equal to or smaller than the image dimensions.")
    
    crop_coordinates = [[x_min, x_max], [y_min, y_max]]
    return crop_coordinates


#helper function
def get_bin_edges(A: list[list[int]], bins):
    """ Get bins_x and bins_y (the bin edges) from the range matrix A ([[xmin, xmax], [ymin, ymax]]) and bins as in the np.histogram2d function.

    Parameters
    ----------
    A : range matrix A, np.ndarray
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.

    Returns
    -------
    bins_x : array
    bins_y : array
    """
    A = np.array(A)

    if isinstance(bins, int):
        bins_x = np.linspace(A[0, 0], A[0, 1], bins+1)
        bins_y = np.linspace(A[1, 0], A[1, 1], bins+1)
    elif isinstance(bins, (list,np.ndarray)) and len(bins) != 2:
        bins_x = bins
        bins_y = bins
    elif isinstance(bins, (list, tuple)) and len(bins) == 2 and all(isinstance(b, int) for b in bins):
        bins_x = np.linspace(A[0, 0], A[0, 1], bins[0]+1)
        bins_y = np.linspace(A[1, 0], A[1, 1], bins[1]+1)
    elif isinstance(bins, (list, tuple)) and len(bins) == 2 and all(isinstance(b, (list, np.ndarray)) for b in bins):
        bins_x = np.array(bins[0])
        bins_y = np.array(bins[1])
    elif isinstance(bins, (list, tuple)) and len(bins) == 2 and isinstance(bins[0], int) and isinstance(bins[1], (list, np.ndarray)):
        bins_x = np.linspace(A[0, 0], A[0, 1], bins[0]+1)
        bins_y = np.array(bins[1])
    elif isinstance(bins, (list, tuple)) and len(bins) == 2 and isinstance(bins[1], int) and isinstance(bins[0], (list, np.ndarray)):
        bins_x = np.array(bins[0])
        bins_y = np.linspace(A[1, 0], A[1, 1], bins[1]+1)
    else:
        raise ValueError("Invalid 'bins' parameter format")

    return bins_x, bins_y

#helper function 
def get_eligible_celltypes(adata_sp: AnnData, 
                           adata_sc: AnnData, 
                           key: str='celltype', 
                           layer: str='lognorm',
                           min_number_cells: int=10):
    """ Get shared celltypes of adata_sp and adata_sc, that have at least min_number_cells members.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data

    Returns
    -------
    celltypes, adata_sp, adata_sc

    """
    # set the layer for adata_sc and adata_sp
    # for most metrics, we use the lognorm layer
    # for negative marker purity, we use the raw layer
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.X = a.X.toarray()

    # take the intersection of genes in adata_sp and adata_sc, as a list
    intersect_genes = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect_genes].copy()
    adata_sp=adata_sp[:,intersect_genes].copy()

    # get the celltypes that are in both adata_sp and adata_sc
    intersect_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[intersect_celltypes]
    celltype_count_sp = adata_sp.obs[key].value_counts().loc[intersect_celltypes]      
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()

    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)].copy()
    adata_sp = adata_sp[adata_sp.obs[key].isin(celltypes)].copy()
    
    return celltypes, adata_sp, adata_sc
