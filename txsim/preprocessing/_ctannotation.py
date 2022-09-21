import numpy as np
import anndata as ad
from anndata import AnnData
import pandas as pd
import scanpy as sc


def run_majority_voting(
    adata_st: AnnData,
    spots: pd.DataFrame
) -> AnnData:

    """Add cell type annotation by majority voting of available annotations listed in ``spots['celltype']``

    Parameters
    ----------
    adata_st : AnnData
        File name of CSV containing genes and cell assignments
    spots : pd.DataFrame
        File name containing cell type for each cell, by default None
        
    Returns
    -------
    AnnData
        Anndata object with cell type annotation in ``adata_st.obs['ct_majority']``
    """

    for cell_id in adata_st.obs['cell_id']:
        cts = spots[spots['cell'] == cell_id ]['Gene'].value_counts()
        
        if 'celltype' in spots.columns:
            mode = spots[spots['cell'] == cell_id ]['celltype'].mode()
            adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_majority'] = mode.values[0]
            adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_majority_cert'] = (spots[spots['cell'] == cell_id ]['celltype'].value_counts()[mode].values[0] / sum(cts))
            
        else:
            print('No celltypes available in spots object')

    return adata_st


# ToDo           
#def run_ssam()

