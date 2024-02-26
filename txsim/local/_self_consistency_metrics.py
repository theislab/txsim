import numpy as np
import pandas as pd
import anndata as ad
import sklearn.metrics
from typing import Tuple

from ._utils import _get_bin_ids

#TODO: "annotation_similarity" 


def _get_ARI_between_cell_assignments_grid(
        adata_sp1: ad.AnnData,
        adata_sp2: ad.AnnData,
        region_range: Tuple[Tuple[float, float], Tuple[float, float]],
        bins: Tuple[int, int],
        uns_key: str = "spots",
        ann_key: str = "cell_id",
        spots_x_col: str = "x",
        spots_y_col: str = "y",
    ) -> np.ndarray:
    ''' Calculate the Adjusted Rand Index (ARI) between two assignments of cell ids to spots for every bin
    
     Parameters
    ----------
    adata_sp1 : AnnData
        Annotated ``AnnData`` object with counts from spatial data and spots from clustering1
    adata_sp2 : AnnData
        Annotated ``AnnData`` object with counts from spatial data and spots from clustering2
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    uns_key : str
        Key where to find the data containing the spots information in both adata.uns
    ann_key : str
        Key where the annotation for the cell IDs are found in adata.uns[uns_key]
    spots_x_col : str, default "x"
        The column name in adata.uns[uns_key] for the x-coordinates of spots. Must be the same for both datasets.
    spots_y_col : str, default "y"
        The column name in adata.uns[uns_key] for the y-coordinates of spots. Must be the same for both datasets.
        
    Returns
    -------
    ARI_per_bins : float
        Adjusted rand Index for every bin
        
    '''
    
    df1 = adata_sp1.uns[uns_key].copy()
    df2 = adata_sp2.uns[uns_key].copy()
    assert (len(df1)==len(df2)), "AnnData Objects do not have the same number of spots."
    
    # get bin ids
    df1 = _get_bin_ids(df1, region_range, bins, x_col=spots_x_col, y_col=spots_y_col)
    df2 = _get_bin_ids(df2, region_range, bins, x_col=spots_x_col, y_col=spots_y_col)
    
    # Set nan to -1 for ARI calculation
    df1.loc[df1[ann_key].isnull(), ann_key] = -1  
    df2.loc[df2[ann_key].isnull(), ann_key] = -1
    
    # group by bins
    df1 = df1.groupby(["y_bin", "x_bin"])[ann_key]
    df2 = df2.groupby(["y_bin", "x_bin"])[ann_key]

    # calculate ARI for every bin
    ARI_per_bin = np.full(bins, np.nan)
    for (y, x), group1 in df1:
        if y == -1 or x == -1:
            continue
        group2 = df2.get_group((y, x))
        c1 = group1.values
        c2 = group2.values
        ARI_per_bin[y, x] = sklearn.metrics.adjusted_rand_score(c1, c2)

    return ARI_per_bin
