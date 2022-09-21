import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from anndata import AnnData

def similar_ge_across_clusters(adata_sp: AnnData, adata_sc: AnnData):
    """Calculate the difference between mean normalized expression of genes across lusters in both modalities
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
    scores : list
       similarity value for every gene across clusters
    """   
    key='celltype'
    adata_sc.X = adata_sc.layers['lognorm']
    adata_sp.X = adata_sp.layers['lognorm']
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
    mean_celltype_sp.loc[:,list(mean_celltype_sp.sum(axis=0)==0)]=0.1
    mean_celltype_sc.loc[:,list(mean_celltype_sc.sum(axis=0)==0)]=0.1
    mean_celltype_sp_norm=mean_celltype_sp.div(mean_celltype_sp.mean(axis=0),axis=1)
    mean_celltype_sc_norm=mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)
    values=np.mean(abs(mean_celltype_sp_norm-mean_celltype_sc_norm),axis=0)
    scores=pd.DataFrame(values,columns=['score']).sort_values(by='score')
    return scores
    
    
def mean_similarity_gene_expression_across_clusters(adata_sp: AnnData, adata_sc: AnnData, pipeline_output:bool=True)-> float:
    """Calculate mean similarity of the difference between mean normalized expression of genes across lusters in both modalities
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
    output_value : float
        mean of of the difference between mean normalized expression of genes across lusters in both modalities
    """   
    scores=similar_ge_across_clusters(adata_sp, adata_sc)
    
    output_value=np.mean(scores)
    if pipeline_output==True:
        return output_value
    else:
        return float(output_value),scores


def median_similarity_gene_expression_across_clusters(adata_sp: AnnData, adata_sc: AnnData, pipeline_output:bool=True)-> float:
    """Calculate meedian value of the similarity of the difference between mean normalized expression of genes across lusters in both modalities
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
    output_value : float
        meedian of of the difference between mean normalized expression of genes across lusters in both modalities
    """   
    scores=similar_ge_across_clusters(adata_sp, adata_sc)
    
    output_value=np.median(scores)
    if pipeline_output==True:
        return output_value
    else:
        return float(output_value),scores



def percentile95_similarity_gene_expression_across_clusters(adata_sp: AnnData, adata_sc: AnnData, pipeline_output:bool=True)-> float:
    """Calculate percentile 95, considering the similarity of the difference between mean normalized expression of genes across lusters in both modalities
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
    output_value : float
        meedian of of the difference between mean normalized expression of genes across lusters in both modalities
    """   
    scores=similar_ge_across_clusters(adata_sp, adata_sc)
    
    output_value=np.percentile(scores,95)
    if pipeline_output==True:
        return output_value
    else:
        return float(output_value),scores
