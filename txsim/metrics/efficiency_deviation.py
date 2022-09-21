import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData


def efficiency_deviation(adata_sp: AnnData, adata_sc: AnnData, pipeline_output:bool=True,key='celltype'):
    """Calculate the efficiency deviation present between the genes in the panel. 
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    efficiency_std : float
       Standard deviation of the calculated efficiencies for every gene. The higher it is, the more different the capture efficiencies are in comparison with the scRNAseq for every gene
    efficiency_mean: float
        Mean efficiency found when comparing scRNAseq and spatial for the overall panel tested
    gene_ratios: pandas dataframe
        Calculated efficiency for every gene in the panel when comparing scRNAseq to spatial
    """   
    adata_sp.X = adata_sp.layers['lognorm']
    adata_sc.X = adata_sp.layers['lognorm']
    adata_sc=adata_sc[:,adata_sc.var['spatial']]
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    genes=adata_sc.var.index[adata_sc.var.index.isin(adata_sp.var.index)]
    exp_sc=pd.DataFrame(adata_sc.X,columns=adata_sc.var.index)
    gene_means_sc=pd.DataFrame(np.mean(exp_sc,axis=0))
    gene_means_sc=gene_means_sc.loc[gene_means_sc.index.sort_values(),:]
    exp_sp=pd.DataFrame(adata_sp.X,columns=adata_sp.var.index)
    gene_means_sp=pd.DataFrame(np.mean(exp_sp,axis=0))
    gene_means_sp=gene_means_sp.loc[gene_means_sp.index.sort_values(),:]
    exp_sc['celltype']=list(adata_sc.obs['celltype'])
    exp_sp['celltype']=list(adata_sp.obs['celltype'])
    exp_sc=exp_sc.loc[exp_sc['celltype'].isin(unique_celltypes),:]
    exp_sp=exp_sp.loc[exp_sp['celltype'].isin(unique_celltypes),:]
    mean_celltype_sp=exp_sp.groupby('celltype').mean()
    mean_celltype_sc=exp_sc.groupby('celltype').mean()
    mean_celltype_sc=mean_celltype_sc.loc[:,mean_celltype_sc.columns.sort_values()]
    mean_celltype_sp=mean_celltype_sp.loc[:,mean_celltype_sp.columns.sort_values()]
    #If no read is prestent in a gene, we add 0.1 so that we can compute statistics
    mean_celltype_sp.loc[:,list(mean_celltype_sp.sum(axis=0)==0)]=0.001
    mean_celltype_sc.loc[:,list(mean_celltype_sc.sum(axis=0)==0)]=0.001
    #mean_celltype_sp_norm=mean_celltype_sp.div(mean_celltype_sp.mean(axis=0),axis=1)
    #mean_celltype_sc_norm=mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)
    gene_ratios=pd.DataFrame(np.mean(mean_celltype_sp,axis=0)/np.mean(mean_celltype_sc,axis=0))
    gr=pd.DataFrame(gene_ratios)
    gr.columns=['efficiency_st_vs_sc']
    efficiency_mean=np.mean(gene_ratios)
    efficiency_std=np.std(gene_ratios)
    if pipeline_output==True:
        return efficiency_std
    else:
        return efficiency_std,efficiency_mean,gr

