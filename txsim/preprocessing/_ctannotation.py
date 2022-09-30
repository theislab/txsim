import numpy as np
import anndata as ad
from anndata import AnnData
import pandas as pd
import scanpy as sc
import plankton.plankton as pl
from plankton.utils import ssam


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
        Anndata object with cell type annotation in ``adata_st.obs['ct_majority']`` and certainty as ``adata_st.obs['ct_majority_cert']`` 
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


        
def run_ssam(
    
    adata_st: AnnData,
    spots: pd.DataFrame,
    adata_sc: AnnData,
    um_p_px: float = 0.325,
    
) -> AnnData:
    """Add cell type annotation by ssam.

    Parameters
    ----------
    coordinates : pd.DataFrame
        DataFrame containing all spots with gene labels and coordinates
    adata_st : AnnData
        AnnData object of the spatial transcriptomics data
    adata_sc : AnnData
        AnnData object of the single cell data
    um_p_px : float
        Conversion factor micrometer per pixel. Adjust to data set
        
    Returns
    -------
    AnnData
        Anndata object with cell type annotation in ``adata_st.obs['ct_ssam']`` and ``adata_st.obs['ct_ssam_cert']``, whereby the latter is the percentage of spots per cell with consistent cell type assignment.
    """
    
    x =  spots.x.values 
    y =  spots.y.values 
    g =  spots.Gene.values
    sdata = pl.SpatialData( spots.Gene,
                            spots.x*um_p_px,
                            spots.y*um_p_px )
    intersect = list(set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_sc=adata_sc[:,intersect]
    exp=pd.DataFrame(adata_sc.X,columns=adata_sc.var_names)
    exp['celltype']=list(adata_sc.obs['celltype'])
    signatures=exp.groupby('celltype').mean().transpose()
    # 'cheat-create' an anndata set:
    adata = AnnData(signatures.T)
    adata.X = np.array(np.nan_to_num(signatures.T))
    adata.obs['celltype'] = adata.obs.index
    # pl.ScanpyDataFrame(sdata,adata)
    sdata = pl.SpatialData(
                            spots.Gene,
                            spots.x*um_p_px,
                            spots.y*um_p_px,
    #                         pixel_maps={'DAPI':bg_map},
                            scanpy=adata
                            )
    sdata = sdata.clean()
    signatures = sdata.scanpy.generate_signatures()
    signatures[signatures.isna()]=0
    #signatures.to_csv('signatures.csv')
    ctmap = ssam(sdata,signatures=signatures,kernel_bandwidth=4,patch_length=1500,threshold_cor=0.2,threshold_exp=0.1)
    # sample the map's values at all molecule locations:
    values_at_xy = ctmap.get_value(sdata.x,sdata.y)
    # assign tissue label to sampled values:
    celltype_labels = adata.obs['celltype'].values[values_at_xy]
    celltype_labels[ctmap.get_value(sdata.x,sdata.y)==-1]='other'
    # add to sdata frame:
    sdata['celltype']= celltype_labels
    sdata['celltype'] = sdata.celltype.astype('category')
    
    # Assign based on majority vote
    spots['celltype'] = sdata['celltype']
    
    for cell_id in adata_st.obs['cell_id']:
        cts = spots[spots['cell'] == cell_id ]['Gene'].value_counts()
        mode = spots[spots['cell'] == cell_id ]['celltype'].mode()
        adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam'] = mode.values[0]
        adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam_cert'] = \
        (spots[spots['cell'] == cell_id ]['celltype'].value_counts()[mode].values[0] / sum(cts))
    
    return adata_st
