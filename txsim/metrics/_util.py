from anndata import AnnData
import numpy as np
# from ._negative_marker_purity import get_spot_assignment_col
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse import isspmatrix 

#TODO: fix Warnings in get_cells_location

#helper function 
def check_crop_exists(x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray):
    """Check if crop coordinates exist.
    
    For this, we check if either (x_min, x_max, y_min, y_max) or an image was provided. If not, we raise a ValueError. 

    Parameters
    ----------
    x_min: int, x_max: int, y_min: int, y_max: int
        crop coordinates
    image: np.ndarray

    Returns
    -------
    if no ValueError was raised, returns range  
    """
    if (x_min is None or x_max is None or y_min is None or y_max is None) and image is None:
        raise ValueError("please provide an image or crop")         
        
    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        range = [[x_min,x_max],[y_min,y_max]]
    
    else:
        range = [[0,image.shape[0]],[0,image.shape[1]]]
        
    return range 

"""TODO: I moved get_cells_location from here back to _negative_marker_purity.py, 
# because it is dependent on the function get_spot_assignment_col
# from _negative_marker_purity.py, which is again dependent on 
# another function from _negative_marker_purity.py: get_negative_marker_dict """
# def get_cells_location(adata_sp: AnnData, adata_sc: AnnData):
#     """Add x,y coordinate columns of cells to adata_sp.obs.

#         Parameters
#         ----------
#         adata_sp : AnnData
#             Annotated ``AnnData`` object with counts from spatial data
#         adata_sc : AnnData
#             Annotated ``AnnData`` object with counts scRNAseq data
#     """

#     get_spot_assignment_col(adata_sp,adata_sc)
#     spots = adata_sp.uns["spots"]
#     df_cells = spots.loc[spots["spot_assignment"]!="unassigned"]      
#     df_cells = df_cells.groupby(["cell"])[["x","y"]].mean()
#     df_cells = df_cells.reset_index().rename(columns={'cell':'cell_id'})
#     adata_sp.obs = pd.merge(df_cells,adata_sp.obs,left_on="cell_id",right_on="cell_id",how="inner")


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
    # # Set threshold parameters 

    # # Liya: "I think min_number_cells should be a parameter of the function, not a global variable. 
    # I added it as a parameter and set default to 10."
    # min_number_cells=10 # minimum number of cells belonging to a cluster to consider it in the analysis

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
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp = adata_sp[adata_sp.obs[key].isin(celltypes)]
    
    return celltypes, adata_sp, adata_sc
