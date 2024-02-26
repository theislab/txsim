import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn.metrics
from anndata import AnnData


def calc_rand_index(assignments: DataFrame):
     """TODO: Exchange with calc_rand_index_NEW"""
     rand_matrix = np.zeros([len(assignments.columns), len(assignments.columns)])
     rand_matrix = pd.DataFrame(rand_matrix)
     for i in range(len(assignments.columns)):
         for j in range(len(assignments.columns)):
             c1 = assignments.iloc[:, i]
             c2 = assignments.iloc[:, j]
             rand_matrix.iloc[i, j] = sklearn.metrics.rand_score(c1,c2)

     rand_matrix.columns = assignments.columns
     rand_matrix.index = assignments.columns

     return rand_matrix


def calc_rand_index_NEW(
    adata_sp1: ad.AnnData,
    adata_sp2: ad.AnnData,
    uns_key: str = "spots",
    ann_key: str = "cell_id",
    pipeline_output: bool=True):
    '''
     Parameters
    ----------
    adata_sp1 : AnnData
        Annotated ``AnnData`` object with counts from spatial data and spots from clustering1
    adata_sp2 : AnnData
        Annotated ``AnnData`` object with counts from spatial data and spots from clustering2
    uns_key : str
        Key where to find the data containing the spots information in both adata.uns
    ann_key : str
        Key where the annotation for teh cell IDs are found in adata.uns[uns_key]
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    rand_index : float
       Increase in proportion of positive cells assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    '''
    assert (len(adata_sp1.uns[uns_key])== len(adata_sp2.uns[uns_key])) , "adatas do not have the same number of spots"

    rand_index = sklearn.metrics.adjusted_rand_score(
        adata_sp1.uns[uns_key][ann_key].values, adata_sp2.uns[uns_key][ann_key].values
    )

    return rand_index

def aggregate_rand_index(matrices: list):
    df_mean = matrices[0].copy()
    df_std = matrices[0].copy()
    df_mean.loc[:,:] = np.mean(np.array( matrices ), axis=0)  
    df_std.loc[:,:] = np.std(np.array( matrices ), axis=0)
    
    return [df_mean, df_std]

def calc_annotation_similarity(adata1: AnnData, adata2: AnnData):
    # Get all the spots assigned to a cell in at least one of the adata objects
    # notnull = assigned to cell
    indices1 = adata1.uns['spots']['cell'][adata1.uns['spots']['cell'].notnull() | adata2.uns['spots']['cell'].notnull()]
    indices2 = adata2.uns['spots']['cell'][adata1.uns['spots']['cell'].notnull() | adata2.uns['spots']['cell'].notnull()]

    #Create temporary copy of cell_id -> celltype dictionary, and add "nan" value
    # for when a transcript (spot) is assigned to a cell in 1 adata, but not the other (cell_id = nan)
    temp_cell_dict2 = adata2.obs[['cell_id','celltype']].set_index('cell_id').loc[:,'celltype'].copy()
    temp_cell_dict2[np.nan] = 'None_1'

    temp_cell_dict1 = adata1.obs[['cell_id','celltype']].set_index('cell_id').loc[:,'celltype'].copy()
    temp_cell_dict1[np.nan] = 'None_2'

    # Use the cell dictionary to get celltypes for each index
    # length of indices1 and 2 are same
    similarity = sum(temp_cell_dict1[indices1].values == temp_cell_dict2[indices2].values) / len(indices1)

    return similarity

def calc_annotation_matrix(adata_list: list, name_list: list):
    ann_matrix = np.zeros([len(adata_list), len(adata_list)])
    ann_matrix = pd.DataFrame(ann_matrix)
    for i in range(len(adata_list)):
        for j in range(len(adata_list)): #TODO Mirror matrix instead of running it twice?
            adata1 = adata_list[i]
            adata2 = adata_list[j]
            ann_matrix.iloc[i, j] = calc_annotation_similarity(adata1, adata2)

    ann_matrix.columns = name_list
    ann_matrix.index = name_list
    
    return ann_matrix
