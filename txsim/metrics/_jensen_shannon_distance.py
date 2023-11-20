import scanpy as sc
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial import distance

def jensen_shannon_distance_metrics(adata_sp: AnnData, adata_sc: AnnData, 
                              key:str='celltype', layer:str='norm', 
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
    # Set threshold parameters
    min_number_cells=20 # minimum number of cells belonging to a cluster to consider it in the analysis

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

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer]= a.layers[layer].toarray()

    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[intersect_celltypes]
    celltype_count_sp = adata_sp.obs[key].value_counts().loc[intersect_celltypes]
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()

    # subset adata_sc and adata_sp to only include eligible celltypes
    adata_sc=adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp=adata_sp[adata_sp.obs[key].isin(celltypes)]

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
        for celltype in celltypes:
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype)
        jsd = sum / n_celltypes
        new_entry = pd.DataFrame([[gene, jsd]],
                   columns=['Gene', 'JSD'])
        per_gene_metric = pd.concat([per_gene_metric, new_entry])
    # set gene as index
    per_gene_metric.set_index('Gene', inplace=True)    
    
    ################
    # PER-CELLTYPE METRIC
    ################
    per_celltype_metric = pd.DataFrame(columns=['celltype', 'JSD'])
    for celltype in set(adata_sp.obs['celltype']):
        sum = 0
        for gene in intersect_genes:
            sum += jensen_shannon_distance_per_gene_and_celltype(adata_sp, adata_sc, gene, celltype)
        jsd = sum / n_genes
        new_entry = pd.DataFrame([[celltype, jsd]],
                     columns=['celltype', 'JSD'])
        per_celltype_metric = pd.concat([per_celltype_metric, new_entry])
    # add the rows with celltypes which were filtered out because of the min_number_cells threshold
    celltypes_with_nan = list(set(intersect_celltypes) - set(celltypes))
    for celltype in celltypes_with_nan:
        new_entry = pd.DataFrame([[celltype, np.nan]],
                     columns=['celltype', 'JSD'])
        per_celltype_metric = pd.concat([per_celltype_metric, new_entry])
    # set celltype as index
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
    sp = np.array(adata_sp[adata_sp.obs['celltype']==celltype][:,gene].X)
    sc = np.array(adata_sc[adata_sc.obs['celltype']==celltype][:,gene].X)
    #flatten the arrays to 1-dim vectors
    sp = sp.flatten().astype(int)
    sc = sc.flatten().astype(int)
    # 1. calculate the distribution vectors for the two datasets
    P, Q = get_probability_distributions_for_sp_and_sc(sp, sc)
    # # 2. if both vectors are empty, return 0
    # if (sum(P) == 0 and sum(Q) == 0):
    #     return 0
    # # 3. if one of the vectors is empty, return 1 (maximum distance)
    # elif (sum(P) == 0 and sum(Q) != 0) or (sum(P) != 0 and sum(Q) == 0):
    #     return 1
    # 4. calculate the Jensen-Shannon distance
    return distance.jensenshannon(P, Q)


def get_probability_distributions_for_sp_and_sc(v_sp:np.array, v_sc:np.array):
    """Calculate the probability distribution vectors from one celltype and one gene
    from spatial and single-cell data
    ---------- 
    v_sp: np.array
        spatial data from one celltype and one gene, 1-dim vector
    v_sc: np.array
        single-cell data from one celltype and one gene, 1-dim vector

    Returns
    -------
    probability_distribution_sp: np.array
        probability distribution from spatial data for one celltype and one gene, 1-dim vector
    probability_distribution_sc: np.array
        probability distribution from dissociated sc data for one celltype and one gene, 1-dim vector
    """
    # find the maximum value in the two given vectors
    max_value = max(max(v_sp), max(v_sc))
    # find the minimum value in the two given vectors
    min_value = min(min(v_sp), min(v_sc))

    # Calculate the histogram
    hist_sp, bin_edges = np.histogram(v_sp, bins=int(max_value - min_value + 1), density=True)
    hist_sc, bin_edges = np.histogram(v_sc, bins=int(max_value - min_value + 1), density=True)
    # hist_sp, bin_edges = np.histogram(v_sp, bins=100, density=True)
    # hist_sc, bin_edges = np.histogram(v_sc, bins=100, density=True)

    # Normalize the histogram to obtain the probability distribution
    d_sp = hist_sp / np.sum(hist_sp)
    d_sc = hist_sc / np.sum(hist_sc)
    return d_sp, d_sc

# TODO: deal with empty expression vectors per gene per celltype
# TODO: if the expression vector is empty, then I get: "FutureWarning: 
# The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. 
# In a future version, this will no longer exclude empty or all-NA columns when 
# determining the result dtypes. To retain the old behavior, exclude the relevant 
# entries before the concat operation.
# per_celltype_metric = pd.concat([per_celltype_metric, new_entry])" - I NEED TO
# build a check for empty vectors, then this concatination won't cause an issue in 
# the future versions
# TODO: filtering out celltypes with less than x cells somehow does not work yet
# TODO: I convert the float values to integers now, change that to use the normalized values


####
