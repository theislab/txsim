import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn.metrics

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