import numpy as np
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse import isspmatrix 


def get_eligible_celltypes(adata_sp: ad.AnnData, 
                           adata_sc: ad.AnnData, 
                           key: str='celltype', 
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