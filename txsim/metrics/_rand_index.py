import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn.metrics
from anndata import AnnData

def calc_rand_index(assignments: DataFrame):
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

def aggregate_rand_index(matrices: list):
    df_mean = matrices[0].copy()
    df_std = matrices[0].copy()
    df_mean.loc[:,:] = np.mean(np.array( matrices ), axis=0)  
    df_std.loc[:,:] = np.std(np.array( matrices ), axis=0)
    
    return [df_mean, df_std]

def calc_annotation_similarity(adata1: AnnData, adata2: AnnData):
    #Lots of indexing- get all the spots assigned to a cell (in at least one of the adata objects)
    #notnull = assigned to cell
    #For these indices, get the cell_id
    indices = adata2.uns['spots']['cell'][adata1.uns['spots']['cell'].notnull() | adata2.uns['spots']['cell'].notnull()]

    #Given the cell_id, get the celltypes of the cell for each spot
    cellann1 = adata1.obs[['cell_id','celltype']].set_index('cell_id').loc[indices.values,'celltype'].values
    #Also for second adata
    cellann2 = adata2.obs[['cell_id','celltype']].set_index('cell_id').loc[indices.values,'celltype'].values

    #calc how many are the same
    similarity = sum(cellann1.astype(str) == cellann2.astype(str)) / len(cellann2)

    return similarity

def calc_annotation_matrix(adata_list: list, name_list: list):
    ann_matrix = np.zeros([len(adata_list), len(adata_list)])
    ann_matrix = pd.DataFrame(ann_matrix)
    for i in range(len(adata_list)):
        for j in range(len(adata_list)):
            adata1 = adata_list[i]
            adata2 = adata_list[j]
            ann_matrix.iloc[i, j] = calc_annotation_similarity(adata1, adata2)

    ann_matrix.columns = name_list
    ann_matrix.index = name_list
    
    return ann_matrix
