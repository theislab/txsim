import scanpy as sc
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial import distance
from scipy.stats import entropy
from numpy.linalg import norm


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def jensen_shannon_divergence(adata_sp: AnnData, adata_sc: AnnData, 
                              key:str='celltype', layer:str='lognorm', 
                              pipeline_output: bool=True):
    """Calculate the Jensen-Shannon divergence between the two distributions:
    the spatial and dissociated single-cell data distributions. Jensen-Shannon
    is an asymmetric metric that measures the relative entropy or difference 
    in information represented by two distributions.
    ----------
    adata_sp: AnnData
        annotated ``AnnData`` object containing the spatial single-cell data
    adata_sc: AnnData
        annotated ``AnnData`` object containing the dissociated single-cell data
    key: str (default: 'celltype')
        .obs column of ``AnnData`` that contains celltype information
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric
    pipeline_output: bool (default: True)
        whether to return only the overall metric (pipeline style)
        (if False, will return the overall metric, per-gene metric and per-celltype metric)

    Returns
    -------
    overall_metric: float
        overall Jensen-Shannon divergence between the two distributions
    per_gene_metric: float
        per gene Jensen-Shannon divergence between the two distributions
    per_celltype_metric: float
        per celltype Jensen-Shannon divergence between the two distributions
    """

    ### SET UP 
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # take the intersection of genes present in adata_sp and adata_sc, as a list
    intersect = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect]
    adata_sp=adata_sp[:,intersect]

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer]= a.layers[layer].toarray()

    ##############################
    #actual calculation of jenson shannon divergence
    

    ##############################

    ####
    #TODO delete this later
    overall_metric = 0
    # per_gene_metric = 0
    # per_celltype_metric = 0
    ####

    if pipeline_output:
         return overall_metric

    return overall_metric, per_gene_metric, per_celltype_metric

####
# #TODO delete this later 
# generate two two-dimensional arrays a and b
a = [[0.1, 0.2, 0.7], 
     [0.3, 0.3, 0.5]]
b = [[0.1, 0.21, 0.7], 
     [0.4, 0.2, 0.5]]
for i in range(len(a)):
    # print(JSD(a[i],b[i])) # calculate the JS divergence between a and b
    print(math.sqrt(JSD(a[i],b[i]))) # apply the square root to the JS divergence to ghe the JS distance
    # print(distance.jensenshannon(a[i],b[i])) # it is the same value as calculated the JS distance using scipy
####