from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from pandas import DataFrame
from ._coexpression_similarity import *
from ._celltype_proportions import *
from ._efficiency import *
from ._relative_pairwise_celltype_expression import *
from ._relative_pairwise_gene_expression import *
from ._negative_marker_purity import *
from ._cell_statistics import *
from ._coembedding import knn_mixing
from ._gene_set_coexpression import *

def all_metrics(
    adata_sp: AnnData,
    adata_sc: AnnData
) -> DataFrame:

    #Generate metrics
    metrics = {}
    ###Celltype proportion
    #metrics['mean_ct_prop_dev'] = mean_proportion_deviation(adata_sp,adata_sc)
    #metrics['prop_noncommon_labels_sc'] = proportion_cells_non_common_celltype_sc(adata_sp,adata_sc)
    #metrics['prop_noncommon_labels_sp'] = proportion_cells_non_common_celltype_sp(adata_sp,adata_sc)
    ### Gene efficiency metrics   
    ##metrics['gene_eff_dev'] = efficiency_deviation(adata_sp,adata_sc)
    ##metrics['gene_eff_mean'] = efficiency_mean(adata_sp,adata_sc)
    #
    ## adata_sc.X does not transform into numpy array... 
    ## weird issue! adata_sc = AnnData(obs=adata_sc.obs,var=adata_sc.var,X=adata_sc.X.toarray()) does the job... but look deeper into this, maybe raise issue
    #### Expression similarity metrics
    ##metrics['relative_sim_across_celltype_overall_metric'] = relative_pairwise_celltype_expression(adata_sp,adata_sc,'celltype','lognorm')
    ##metrics['relative_sim_across_gene_overall_metric'] = relative_pairwise_gene_expression(adata_sp, adata_sc, 'celltype', 'lognorm')
    ###metrics['mean_sim_across_clust'] = mean_similarity_gene_expression_across_clusters(adata_sp,adata_sc)
    ###metrics['prc95_sim_across_clust'] = percentile95_similarity_gene_expression_across_clusters(adata_sp,adata_sc)
    #
    ### Coexpression similarity
    #metrics['coexpression_similarity'] = coexpression_similarity(adata_sp, adata_sc)
    #metrics['coexpression_similarity_celltype'] = coexpression_similarity(adata_sp, adata_sc, by_celltype = True)
    ##metrics['gene_set_coexpression'] = gene_set_coexpression(adata_sp, adata_sc)
        
    # Negative marker purity
    metrics['neg_marker_purity_cells'] = negative_marker_purity_cells(adata_sp.copy(),adata_sc.copy())
    metrics['neg_marker_purity_reads'] = negative_marker_purity_reads(adata_sp.copy(),adata_sc.copy())
    
    #### KNN mixing
    #metrics['knn_mixing'] = knn_mixing(adata_sp.copy(),adata_sc.copy())
    
    ## Cell statistics
    #metrics['ratio_median_readsxcell'] = ratio_median_readsXcells(adata_sp,adata_sc)
    #metrics['ratio_mean_readsxcell'] = ratio_mean_readsXcells(adata_sp,adata_sc)
    #metrics['ratio_n_cells'] = ratio_number_of_cells(adata_sp,adata_sc)
    #metrics['ratio_mean_genexcells'] = ratio_mean_genesXcells(adata_sp,adata_sc)
    #metrics['ratio_median_genexcells'] = ratio_median_genesXcells(adata_sp,adata_sc)
        
    
    return pd.DataFrame.from_dict(metrics, orient='index')

def aggregate_metrics(
    metric_list: list,
    aggregated_metric: DataFrame,
    name_list: list = None
):
    mean_metric = pd.concat((metric_list), axis=1)
    if name_list is not None: mean_metric.columns = name_list
    mean_metric["mean"] = mean_metric.mean(axis=1)
    mean_metric["std"] = mean_metric.std(axis=1)
    aggregated_metric.columns = ["AGGREGATED_METRIC"]
    mean_metric = pd.concat((mean_metric,aggregated_metric), axis=1)
    return mean_metric
