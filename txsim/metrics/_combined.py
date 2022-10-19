from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from ._coexpression_similarity import *
from .metric2 import calc_metric2
from ._celltype_proportions import *
from ._efficiency import *
from ._expression_similarity_between_celltypes import *
from ._negative_marker_purity import *
from ._cell_statistics import *
from ._coembedding import knn_mixing
from ._gene_set_coexpression import *


def all_metrics(
    adata_sp: AnnData,
    adata_sc: AnnData
) -> pd.DataFrame:

    #Generate metrics
    metrics = {}
    #Celltype proportion
    metrics['mean_ct_prop_dev'] = mean_proportion_deviation(adata_sp,adata_sc)
    metrics['prop_noncommon_labels_sc'] = proportion_cells_non_common_celltype_sc(adata_sp,adata_sc)
    metrics['prop_noncommon_labels_sp'] = proportion_cells_non_common_celltype_sp(adata_sp,adata_sc)
    # Gene efficiency metrics   
    metrics['gene_eff_dev'] = efficiency_deviation(adata_sp,adata_sc)
    metrics['gene_eff_mean'] = efficiency_mean(adata_sp,adata_sc)
    # Expression similarity metrics
    metrics['mean_sim_across_clust'] = mean_similarity_gene_expression_across_clusters(adata_sp,adata_sc)
    metrics['prc95_sim_across_clust'] = percentile95_similarity_gene_expression_across_clusters(adata_sp,adata_sc)
    # Coexpression similarity
    metrics['coexpression_similarity'] = coexpression_similarity(adata_sp, adata_sc)
    metrics['coexpression_similarity_celltype'] = coexpression_similarity(
        adata_sp,
        adata_sc,
        by_celltype = True)
    metrics['gene_set_coexpression'] = gene_set_coexpression(adata_sp, adata_sc)
    # Negative marker purity
    metrics['neg_marker_purity'] = negative_marker_purity(adata_sp,adata_sc)
    # Cell statistics
    metrics['ratio_median_readsxcell'] = ratio_median_readsXcells(adata_sp,adata_sc)
    metrics['ratio_mean_readsxcell'] = ratio_mean_readsXcells(adata_sp,adata_sc)
    metrics['ratio_n_cells'] = ratio_number_of_cells(adata_sp,adata_sc)
    metrics['ratio_mean_genexcells'] = ratio_mean_genesXcells(adata_sp,adata_sc)
    metrics['ratio_median_genexcells'] = ratio_median_genesXcells(adata_sp,adata_sc)
    # KNN mixing
    metrics['knn_mixing'] = knn_mixing(adata_sp,adata_sc)

    
    
    
    return pd.DataFrame.from_dict(metrics, orient='index')

