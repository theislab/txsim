import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
import math

def similar_ge_across_clusters(adata_sp: AnnData, adata_sc: AnnData, layer: str='lognorm'):
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

    # Set layer to calculate the metric on
    adata_sc.X = adata_sc.layers[layer]
    adata_sp.X = adata_sp.layers[layer]

    # Get shared genes between sc and spatial data
    intersect = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))

    # Subset sc data to spatial genes
    adata_sc=adata_sc[:,intersect]

    # Sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.X = a.X.toarray()
    
    # Get cell types that occur in both sc and spatial
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
        
    # Set gene expression Dataframe for scRNAseq
    exp_sc=pd.DataFrame(adata_sc.X,columns=adata_sc.var.index) # is it needed to get Dataframes

    # Set Dataframe with the mean expression values for each gene across all the cells
    gene_means_sc=pd.DataFrame(np.mean(exp_sc,axis=0))
    # Sort 'gene_means_sc':  the gene names in ascending order
    gene_means_sc=gene_means_sc.loc[gene_means_sc.index.sort_values(),:]

    # Same process for spatial data (Gene expression Dataframe with mean expression values for each gene across all the cells)
    exp_sp=pd.DataFrame(adata_sp.X,columns=adata_sp.var.index)
    gene_means_sp=pd.DataFrame(np.mean(exp_sp,axis=0))
    gene_means_sp=gene_means_sp.loc[gene_means_sp.index.sort_values(),:]
    
    # Add a new column for 'celltype'to the gene expression Dataframes(each spatial and scRNAseq)
    exp_sc['celltype']=list(adata_sc.obs['celltype'])
    exp_sp['celltype']=list(adata_sp.obs['celltype'])

    # Subset gene expression Dataframe(for each scRNAseq and spatial data) including only the unique cell types 
    exp_sc=exp_sc.loc[exp_sc['celltype'].isin(unique_celltypes),:]
    exp_sp=exp_sp.loc[exp_sp['celltype'].isin(unique_celltypes),:]

    # Set Dataframe with corresponding mean expression value each scRNAseq and spatial 
    mean_celltype_sc=exp_sc.groupby('celltype').mean()
    mean_celltype_sc=mean_celltype_sc.loc[:,mean_celltype_sc.columns.sort_values()]
    mean_celltype_sp=exp_sp.groupby('celltype').mean()
    mean_celltype_sp=mean_celltype_sp.loc[:,mean_celltype_sp.columns.sort_values()]

# UPDATED PART WITH A OUR old SUGGESTION
   
    pairwise_differences_per_gene_sc = {}

    # Calculate the pairwise difference between celltypes for each column(gene) for sc
    pair_idx = 0
    for i, col_i in enumerate(mean_celltype_sc.columns, start=0):
        for j, col_j in enumerate(mean_celltype_sc.columns):
            if i < j:
                pairwise_differences_per_gene_sc[pair_idx] = mean_celltype_sc[col_i] - mean_celltype_sc[col_j]
                pair_idx += 1

    pairwise_differences_per_gene_sp = {}

    # Calculate the pairwise difference between celltypes for each column(gene) for spatial
    # numpy-broadcasting - see new_expression_similarity_between_celltypes for new version
    pair_idx = 0
    for i, col_i in enumerate(mean_celltype_sp.columns, start=0):
        for j, col_j in enumerate(mean_celltype_sp.columns):
            if i < j:
                pairwise_differences_per_gene_sp[pair_idx] = mean_celltype_sp[col_i] - mean_celltype_sp[col_j]
                pair_idx += 1

    sc_values =  pd.DataFrame(np.array(list(pairwise_differences_per_gene_sc.values())))
    sp_values =  pd.DataFrame(np.array(list(pairwise_differences_per_gene_sp.values())))

    
    sp_norm=sp_values.div(sp_values.mean(axis=0),axis=1)
    sc_norm=sc_values.div(sc_values.mean(axis=0),axis=1)

    final_difference_abs_values = np.mean(abs(sc_norm - sp_norm),axis=0)
    scores_new =pd.DataFrame(final_difference_abs_values,columns=['score']).sort_values(by='score')

#----------------BELOW IS THE PART WITHOUT CHANGE

    #If no read is prestent in a gene, we add 0.1 so that we can compute statistics
    mean_celltype_sp.loc[:,list(mean_celltype_sp.sum(axis=0)==0)]=0.1
    mean_celltype_sc.loc[:,list(mean_celltype_sc.sum(axis=0)==0)]=0.1

    # Set Dataframes with mean-normalized expression values for each gene
    # what it does: mean of each column of the mean_celltype_sp dataframe along the vertical axis (axis=0), and then divide each value in mean_celltype_sp by the corresponding mean value. 
    # Problem is that mean_celltype_sp has mean values higher than the mean value of the whole gene
    mean_celltype_sp_norm=mean_celltype_sp.div(mean_celltype_sp.mean(axis=0),axis=1) # output has not always absolute values! Reason for no absolute values in scores
    mean_celltype_sc_norm=mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)

    # Create Array with mean absolute difference between the normalized gene expression values for each gene and each cell type
    values=np.mean(abs(mean_celltype_sp_norm-mean_celltype_sc_norm),axis=0)

    # Set Dataframe with one column and rows as 'values'
    scores=pd.DataFrame(values,columns=['score']).sort_values(by='score')
    return scores, mean_celltype_sc


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
