from anndata import AnnData
import numpy as np
import pandas as pd
from ._coexpression_similarity import *

def all_metrics(
    spatial_data: AnnData,
    seq_data: AnnData
) -> pd.DataFrame:

    #Calculate some basic statistics
    spatial_data.obs['n_counts']= np.sum(spatial_data.layers['raw_counts'], axis = 1)
    spatial_data.obs['n_unique_genes']= np.sum(spatial_data.layers['raw_counts']>0, axis = 1)
    spatial_data.var['n_counts']= np.sum(spatial_data.layers['raw_counts'], axis=0)
    spatial_data.var['n_unique_cells']= np.sum(spatial_data.layers['raw_counts']>0, axis = 0)
    
    #Generate metrics
    metrics = {}
    metrics['coex_all_normsc_normst'] = coexpression_similarity(spatial_data, seq_data)
    metrics['coex_thresh_normsc_normst'] = coexpression_similarity(spatial_data, seq_data, thresh=0.5)

    metrics['coex_all_rawsc_normst'] = coexpression_similarity(spatial_data, seq_data, norm_sc=False)
    metrics['coex_thresh_rawsc_normst'] = coexpression_similarity(spatial_data, seq_data, thresh=0.5, norm_sc=False)

    metrics['coex_all_normsc_rawst'] = coexpression_similarity(spatial_data, seq_data, raw=True)
    metrics['coex_thresh_normsc_rawst'] = coexpression_similarity(spatial_data, seq_data, thresh=0.5, raw=True)

    metrics['coex_all_rawsc_rawst'] = coexpression_similarity(spatial_data, seq_data, raw=True, norm_sc=False)
    metrics['coex_thresh_rawsc_rawst'] = coexpression_similarity(spatial_data, seq_data, thresh=0.5, raw=True, norm_sc=False)

    ct =  coexpression_similarity_celltype(spatial_data, seq_data, thresh=0.5)
    if len(ct > 0):
        metrics['coex_bytype_thresh'] = np.nanmean(ct['mean_diff'])
        idx = ~np.isnan(ct['mean_diff'])
        metrics['coex_bytype_weighted_thresh'] = np.average(ct['mean_diff'][idx], weights = ct['pct'][idx])

    metrics['pct_spots_unassigned'] = spatial_data.uns['pct_noise']
    metrics['n_cells'] = spatial_data.n_obs
    metrics['mean_cts_per_cell'] = np.mean(spatial_data.obs['n_counts'])
    metrics['mean_genes_per_cell'] = np.mean(spatial_data.obs['n_unique_genes'])
    metrics['mean_cts_per_gene'] = np.mean(spatial_data.var['n_counts'])
    metrics['mean_cells_per_gene'] = np.mean(spatial_data.var['n_unique_cells'])
    metrics['pct_cells_no_type'] = spatial_data.obs['celltype'].value_counts()['None'] / spatial_data.n_obs
    
    return pd.DataFrame.from_dict(metrics, orient='index')

