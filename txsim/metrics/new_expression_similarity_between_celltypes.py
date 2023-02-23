import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

def relative_celltype_expression(adata_sp: AnnData, adata_sc: AnnData, key:str='celltype', layer:str='lognorm'):
    """Calculate the efficiency deviation present between the genes in the panel. 
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    key: str (default: 'celltype')
        .obs column of ``AnnData`` that contains celltype information
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric

    Returns
    -------
    overall_metric: float
        similarity of relative gene expression across all genes and celltypes, b/t the scRNAseq and spatial data
    per_gene_metric: float
        similarity of relative gene expression per gene across all celltypes, b/t the scRNAseq and spatial data
    per_celltype_metric: float
        similarity of relative gene expression per celltype across all genes, b/t the scRNAseq and spatial data
  
    """   
    ### SET UP
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]
    
    # take the intersection of genes in adata_sp and adata_sc, as a list
    intersect = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))
    
    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect]
    adata_sp=adata_sp[:,intersect]
    
    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.X = a.X.toarray()
            
    # find the unique celltypes in adata_sc that are also in adata_sp
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    
    
    #### FIND MEAN GENE EXPRESSION PER CELL TYPE FOR EACH MODALITY
    # get the adata_sc cell x gene matrix as a pandas dataframe (w gene names as column names)
    exp_sc=pd.DataFrame(adata_sc.X,columns=adata_sc.var.index)
    
    # get the adata_sp cell x gene matrix as a pandas dataframe (w gene names as column names)
    exp_sp=pd.DataFrame(adata_sp.X,columns=adata_sp.var.index)
    
    # add "celltype" label column to exp_sc & exp_sp cell x gene matrices 
    exp_sc[key]=list(adata_sc.obs[key])
    exp_sp[key]=list(adata_sp.obs[key])
    
    # delete all cells from the exp matrices if they aren't in the set of intersecting celltypes b/t sc & sp data
    exp_sc=exp_sc.loc[exp_sc[key].isin(unique_celltypes),:]
    exp_sp=exp_sp.loc[exp_sp[key].isin(unique_celltypes),:]
    
    # find the mean expression for each gene for each celltype in sc and sp data
    mean_celltype_sp=exp_sp.groupby(key).mean()
    mean_celltype_sc=exp_sc.groupby(key).mean()
    
    # sort genes in alphabetical order 
    mean_celltype_sc=mean_celltype_sc.loc[:,mean_celltype_sc.columns.sort_values()]
    mean_celltype_sp=mean_celltype_sp.loc[:,mean_celltype_sp.columns.sort_values()]
    
    
    #### CALCULATE PAIRWISE RELATIVE DISTANCES BETWEEN CELL TYPES
    mean_celltype_sc_np = mean_celltype_sc.T.to_numpy()
    pairwise_distances_sc = mean_celltype_sc_np[:,:,np.newaxis] - mean_celltype_sc_np[:,np.newaxis,:]
    pairwise_distances_sc = pairwise_distances_sc.transpose((1,2,0)) #results in np.array of dimensions (num_genes, num_genes, num_celltypes) 
       
    mean_celltype_sp_np = mean_celltype_sp.T.to_numpy()
    pairwise_distances_sp = mean_celltype_sp_np[:,:,np.newaxis] - mean_celltype_sp_np[:,np.newaxis,:]
    pairwise_distances_sp = pairwise_distances_sp.transpose((1,2,0)) #results in np.array of dimensions (num_genes, num_genes, num_celltypes) 
    
    #### NORMALIZE THESE PAIRWISE DISTANCES BETWEEN CELL TYPES
    #calculate sum of absolute distances
    abs_diff_sc = np.absolute(pairwise_distances_sc.T)
    abs_diff_sum_sc = np.sum(abs_diff_sc, axis=(0,1))
    
    abs_diff_sp = np.absolute(pairwise_distances_sp.T)
    abs_diff_sum_sp = np.sum(abs_diff_sp, axis=(0,1))
    
    # calculate normalization factor
    norm_factor_sc = mean_celltype_sc.T.shape[1]**2 * abs_diff_sum_sc
    norm_factor_sp = mean_celltype_sc.T.shape[1]**2 * abs_diff_sum_sp
    
    #perform normalization
    norm_pairwise_distances_sc = np.divide(pairwise_distances_sc.T, norm_factor_sc)
    norm_pairwise_distances_sp = np.divide(pairwise_distances_sp.T, norm_factor_sp)
    
    
    ##### CALCULATE OVERALL SCORE,PER-GENE SCORES, PER-CELLTYPE SCORES
    overall_score = np.sum(np.absolute(norm_pairwise_distances_sp - norm_pairwise_distances_sc), axis=None)
    overall_metric = 1 - (overall_score/(2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=None)))
    
    per_gene_score = np.sum(np.absolute(norm_pairwise_distances_sp - norm_pairwise_distances_sc), axis=(1,2))
    per_gene_metric = 1 - (per_gene_score/(2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=(1,2))))
    per_gene_metric = pd.DataFrame(per_gene_metric, index=mean_celltype_sc.T.columns, columns=['score']) #add back the gene labels 
    
    per_celltype_score = np.sum(np.absolute(norm_pairwise_distances_sp - norm_pairwise_distances_sc), axis=(0,1))
    per_celltype_metric = 1 - (per_celltype_score/(2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=(0,1))))
    per_celltype_metric = pd.DataFrame(per_celltype_metric, index=mean_celltype_sc.T.index, columns=['score']) #add back the celltype labels 
    
    return overall_metric, per_gene_metric, per_celltype_metric
    