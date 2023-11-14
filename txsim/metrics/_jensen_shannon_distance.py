import scanpy as sc
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial import distance

def jensen_shannon_distance_metrics(adata_sp: AnnData, adata_sc: AnnData, 
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
    intersect_genes = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))
    intersect_celltypes = list(set(adata_sp.obs[key]).intersection(set(adata_sc.obs[key])))

    # number of celltypes
    n_celltypes = len(intersect_celltypes)

    # number of genes
    n_genes = len(intersect_genes)

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect_genes]
    adata_sp=adata_sp[:,intersect_genes]

    # subset adata_sc and adata_sp to only include celltypes in the intersection of adata_sp and adata_sc
    adata_sc=adata_sc[adata_sc.obs[key].isin(intersect_celltypes)]
    adata_sp=adata_sp[adata_sp.obs[key].isin(intersect_celltypes)]

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer]= a.layers[layer].toarray()

    ################
    # OVERALL METRIC
    ################
    overall_metric = 0
    for gene in adata_sc.var_names:
        sum = 0
        for celltype in set(adata_sp.obs['celltype']):
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype)
        overall_metric += sum
    overall_metric = overall_metric / (n_celltypes * n_genes)

    if pipeline_output: # the execution stops here if pipeline_output=True
         return overall_metric

    ################
    # PER-GENE METRIC
    ################
    per_gene_metric = pd.DataFrame(columns=['Gene', 'JSD'])
    for gene in adata_sc.var_names:
        sum = 0
        for celltype in set(adata_sp.obs['celltype']):
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype)
        jsd = sum / n_celltypes
        new_entry = pd.DataFrame([[gene, jsd]],
                   columns=['Gene', 'JSD'])
        per_gene_metric = pd.concat([per_gene_metric, new_entry])
    per_gene_metric.set_index('Gene', inplace=True)    
    ################
    # PER-CELLTYPE METRIC
    ################
    per_celltype_metric = pd.DataFrame(columns=['celltype', 'JSD'])
    for celltype in set(adata_sp.obs['celltype']):
        sum = 0
        for gene in adata_sc.var_names:
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype)
        jsd = sum / n_genes
        new_entry = pd.DataFrame([[celltype, jsd]],
                     columns=['celltype', 'JSD'])
        per_celltype_metric = pd.concat([per_celltype_metric, new_entry])
    per_celltype_metric.set_index('celltype', inplace=True)
    ################
    return overall_metric, per_gene_metric, per_celltype_metric

def jensen_shannon_distance_per_gene_and_celltype(adata_sp:AnnData, adata_sc:AnnData, gene:str, celltype:str):
    """Calculate the Jensen-Shannon distance between two distributions:
    1. expression values for a given gene in a given celltype from spatial data
    2. expression values for a given gene in a given celltype from single-cell data
    ----------
    adata_sp: AnnData
        annotated ``AnnData`` object containing the spatial single-cell data
    adata_sc: AnnData
        annotated ``AnnData`` object containing the dissociated single-cell data
    gene: str
        gene of interest
    celltype: str
        celltype of interest
        
    Returns
    -------
    jsd: float
        Jensen-Shannon distance between the two distributions, single value
    """
    # 0. get all expression values for the gene of interest for the celltype of interest
    P = np.array(adata_sp[adata_sp.obs['celltype']==celltype][:,gene].X)
    Q = np.array(adata_sc[adata_sc.obs['celltype']==celltype][:,gene].X)
    # 1. append the shorter vector with average values
    P = np.squeeze(P) # make sure the vector is 1D
    Q = np.squeeze(Q) # make sure the vector is 1D
    length_difference = abs(len(P) - len(Q))
    if len(P) > len(Q):
        average_values_to_add = np.empty(length_difference)
        average_values_to_add.fill(np.average(Q))
        Q_extended = np.append(Q, average_values_to_add)
        P_extended = P
    elif len(Q) > len(P):
        average_values_to_add = np.empty(length_difference)
        average_values_to_add.fill(np.average(P))
        P_extended = np.append(P, average_values_to_add)
        Q_extended = Q
    else:
        P_extended = P
        Q_extended = Q
    # 2. normalize the vectors
    # TODO # delete # actually this step is not necessary because the scipy function does it for you
    P_normalized = P_extended / np.sum(P_extended)
    Q_normalized = Q_extended / np.sum(Q_extended)
    # 3. calculate the Jensen-Shannon distance
    return distance.jensenshannon(P_normalized, Q_normalized)

# TODO: deal with empty expression vectors
# TODO: if the expression vector is empty, then I get: "FutureWarning: 
# The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. 
# In a future version, this will no longer exclude empty or all-NA columns when 
# determining the result dtypes. To retain the old behavior, exclude the relevant 
# entries before the concat operation.
# per_celltype_metric = pd.concat([per_celltype_metric, new_entry])" - I NEED TO
# build a check for empty vectors, then this concatination won't cause an issue in 
# the future versions


####
