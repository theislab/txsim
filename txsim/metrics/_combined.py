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
    adata_sc: AnnData,
    key: str = 'celltype',
    raw_layer: str = 'raw',
    lognorm_layer: str = 'lognorm'
) -> DataFrame:

    # Generate metrics
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
    metrics['rel_pairwise_ct_expr_sim'] = relative_pairwise_celltype_expression(adata_sp.copy(), adata_sc.copy(), key=key, layer=lognorm_layer)
    metrics['rel_pairwise_gene_expr_sim'] = relative_pairwise_gene_expression(adata_sp.copy(), adata_sc.copy(), key=key, layer=lognorm_layer)
    #
    ### Coexpression similarity
    metrics['coexpr_similarity'] = coexpression_similarity(adata_sp.copy(), adata_sc.copy(), key=key, layer=lognorm_layer)
    metrics['coexpr_similarity_celltype'] = coexpression_similarity(adata_sp.copy(), adata_sc.copy(), by_celltype = True, key=key, layer=lognorm_layer)
    ##metrics['gene_set_coexpression'] = gene_set_coexpression(adata_sp, adata_sc)
        
    # Negative marker purity
    metrics['neg_marker_purity_cells'] = negative_marker_purity_cells(adata_sp.copy(),adata_sc.copy(), key=key, layer=raw_layer)
    metrics['neg_marker_purity_reads'] = negative_marker_purity_reads(adata_sp.copy(),adata_sc.copy(), key=key, layer=raw_layer)
    
    #### KNN mixing
    metrics['knn_mixing'] = knn_mixing(adata_sp.copy(),adata_sc.copy(), key=key, layer=lognorm_layer)
    
    # Cell statistics
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

def aggregate_group_metrics(
    metric_list: list,
    aggregated_metric: DataFrame,
    name_list: list = None
):
    # Hocus pocus required because sometimes the dataframes switch the levels for the MultiIndex
    # I do not know if there is a better way to do this
    # I think it is unnecessary now since the table .csv, but leaving it here
    # uni_index = metric_list[0].index #Set of correctly ordered indices
    # for df in metric_list + [aggregated_metric]: 
    #     for i in range(len(df.index)):
    #         if df.index[i] not in uni_index:
    #             print("here")
    #             pair = df.index[i]
    #             df.reset_index(inplace=True)
    #             df.loc[i, 'run1'] = pair[1]
    #             df.loc[i, 'run2'] = pair[0]
    #             df.set_index(['run1','run2'], inplace=True)
    
    #combine and fix the names
    mean_metric = pd.concat((metric_list), axis=1)
    if name_list is not None: mean_metric.columns = name_list 
    metric_types = {name.split('-')[-1] for name in mean_metric.columns}

    #find mean and std
    for m in metric_types:
        mean_metric["mean-"+m] = mean_metric[[x for x in mean_metric.columns if m in x]].mean(axis=1)
        mean_metric["std-"+ m] = mean_metric[[x for x in mean_metric.columns if (m in x and "mean-" not in x)]].std(axis=1)
    
    #add in aggregated and return
    aggregated_metric.columns = ["aggregated-"+x for x in aggregated_metric.columns]
    mean_metric = pd.concat((mean_metric,aggregated_metric), axis=1)
    return mean_metric