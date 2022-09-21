import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData

def mean_proportion_deviation(adata_sp: AnnData, adata_sc: AnnData,pipeline_output=True):
    """Calculate the mean difference in proportions between cell types present in both datasets
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    mean_proportion_deviation: float
        Mean difference in cell type proportions between cell types present in both datasets. Values close to 0 indicate 
        good consistency in the proportions. Values close to 1 indicate inconsistency between cell type proportions
    sp_minus_sc_proportions: DataFrame
        Dataframe containing the difference in proportions between cell types present in both modalities
    """   

    key='celltype'
    adata_sc = adata_sc[:,adata_sc.var_names]
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    celltype_info_sc=adata_sc.obs.loc[adata_sc.obs[key].isin(unique_celltypes),:]
    celltype_info_sp=adata_sp.obs.loc[adata_sp.obs[key].isin(unique_celltypes),:]
    counts_sc=[]
    counts_sp=[]
    for ct in unique_celltypes:
        counts_sc.append(np.sum(celltype_info_sc[key]==ct))
        counts_sp.append(np.sum(celltype_info_sp[key]==ct))
    counts_common_celltypes=pd.DataFrame([counts_sc,counts_sp],columns=unique_celltypes,index=['sc','spatial']).transpose()
    adata_sc.obs.shape[0]
    ct_proportions=counts_common_celltypes.div([adata_sc.obs.shape[0],adata_sp.obs.shape[0]],axis=1)
    mean_proportion_deviation=np.mean(abs(ct_proportions['sc']-ct_proportions['spatial']))
    if pipeline_output==True:
        return mean_proportion_deviation
    if pipeline_output==False:
        sp_minus_sc_proportions=pd.DataFrame(sp_minus_sc_proportions,columns=['sp-sc_proportions'])
        return mean_proportion_deviation,sp_minus_sc_proportions




def proportion_cells_non_common_celltype_sc(adata_sp: AnnData, adata_sc: AnnData,pipeline_output=True):
    """Returns the proportion of cells in the scRNAseq dataset that are assigned to a cell type not present in spatial dataset
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    proportion_noncommon_ct: float
        proportion of cells in the sc dataset that are assigned to a cell type not present in scRNAseq
    """   

    key='celltype'
    adata_sc = adata_sc[:,adata_sc.var_names]
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    celltype_info_sc=adata_sc.obs.loc[adata_sc.obs[key].isin(unique_celltypes),:]
    celltype_info_sp=adata_sp.obs.loc[adata_sp.obs[key].isin(unique_celltypes),:]
    counts_sc=[]
    counts_sp=[]
    for ct in unique_celltypes:
        counts_sc.append(np.sum(celltype_info_sc[key]==ct))
        counts_sp.append(np.sum(celltype_info_sp[key]==ct))
    counts_common_celltypes=pd.DataFrame([counts_sc,counts_sp],columns=unique_celltypes,index=['sc','spatial']).transpose()
    ct_proportions=counts_common_celltypes.div([adata_sc.obs.shape[0],adata_sp.obs.shape[0]],axis=1)
    proportion_noncommon_ct_sc=1-ct_proportions['sc'].sum()
    return proportion_noncommon_ct_sc



def proportion_cells_non_common_celltype_sp(adata_sp: AnnData, adata_sc: AnnData,pipeline_output=True):
    """Returns the proportion of cells in the spatial dataset that are assigned to a cell type not present in scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    proportion_noncommon_ct: float
        proportion of cells in the spatial dataset that are assigned to a cell type not present in scRNAseq
    """   

    key='celltype'
    adata_sc = adata_sc[:,adata_sc.var_names]
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    celltype_info_sc=adata_sc.obs.loc[adata_sc.obs[key].isin(unique_celltypes),:]
    celltype_info_sp=adata_sp.obs.loc[adata_sp.obs[key].isin(unique_celltypes),:]
    counts_sc=[]
    counts_sp=[]
    for ct in unique_celltypes:
        counts_sc.append(np.sum(celltype_info_sc[key]==ct))
        counts_sp.append(np.sum(celltype_info_sp[key]==ct))
    counts_common_celltypes=pd.DataFrame([counts_sc,counts_sp],columns=unique_celltypes,index=['sc','spatial']).transpose()
    ct_proportions=counts_common_celltypes.div([adata_sc.obs.shape[0],adata_sp.obs.shape[0]],axis=1)
    proportion_noncommon_ct_sp=1-ct_proportions['spatial'].sum()
    return proportion_noncommon_ct_sp


