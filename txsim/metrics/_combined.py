from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import glob as glob
import re as re
import os as os
from os import listdir
from os.path import isfile, join
from _coexpression_similarity import *
#from metric2 import calc_metric2
#from _celltype_proportions import *
from _efficiency import *
from new_expression_similarity_between_celltypes import *
#from _expression_similarity_between_celltypes import *
from _negative_marker_purity import *
#from _cell_statistics import *
from _coembedding import knn_mixing_per_ctype
#from _gene_set_coexpression import *


def all_metrics(
    adata_sp: AnnData,
    adata_sc: AnnData
) -> pd.DataFrame:

    #Generate metrics
    metrics = {}
    ##Celltype proportion
    #metrics['mean_ct_prop_dev'] = mean_proportion_deviation(adata_sp,adata_sc)
    #metrics['prop_noncommon_labels_sc'] = proportion_cells_non_common_celltype_sc(adata_sp,adata_sc)
    #metrics['prop_noncommon_labels_sp'] = proportion_cells_non_common_celltype_sp(adata_sp,adata_sc)
    ## Gene efficiency metrics   
    metrics['relative_sim_across_genes'] = relative_gene_expression(adata_sp,adata_sc,'celltype','lognorm')
    #metrics['gene_eff_dev'] = efficiency_deviation(adata_sp,adata_sc)
    #metrics['gene_eff_mean'] = efficiency_mean(adata_sp,adata_sc)
    ## Expression similarity metrics
    metrics['relative_sim_across_celltype_overall_metric'] = relative_celltype_expression(adata_sp,adata_sc,'celltype','lognorm')
    #metrics['relative_sim_across_celltype_overall_metric'] = all_scores[0]
    #metrics['relative_sim_across_celltype_per_gene_metric'] = all_scores[1]
    #metrics['relative_sim_across_celltype_per_celltype_metric'] = all_scores[2]
    #metrics['mean_sim_across_clust'] = mean_similarity_gene_expression_across_clusters(adata_sp,adata_sc)
    #metrics['prc95_sim_across_clust'] = percentile95_similarity_gene_expression_across_clusters(adata_sp,adata_sc)
    ## Coexpression similarity
    metrics['coexpression_similarity'] = coexpression_similarity(adata_sp, adata_sc)
    #metrics['coexpression_similarity_celltype'] = coexpression_similarity(
    #    adata_sp,
    #    adata_sc,
    #    by_celltype = True)
    #metrics['gene_set_coexpression'] = gene_set_coexpression(adata_sp, adata_sc)
    # Negative marker purity
    metrics['neg_marker_purity_cells'] = negative_marker_purity_cells(adata_sp,adata_sc)
    metrics['neg_marker_purity_reads'] = negative_marker_purity_reads(adata_sp,adata_sc)
    ## Cell statistics
    #metrics['ratio_median_readsxcell'] = ratio_median_readsXcells(adata_sp,adata_sc)
    #metrics['ratio_mean_readsxcell'] = ratio_mean_readsXcells(adata_sp,adata_sc)
    #metrics['ratio_n_cells'] = ratio_number_of_cells(adata_sp,adata_sc)
    #metrics['ratio_mean_genexcells'] = ratio_mean_genesXcells(adata_sp,adata_sc)
    #metrics['ratio_median_genexcells'] = ratio_median_genesXcells(adata_sp,adata_sc)
    ## KNN mixing
    metrics['knn_mixing'] = knn_mixing_per_ctype(adata_sp,adata_sc)

    return pd.DataFrame.from_dict(metrics, orient='index')

# panel_to_celltype =  {'Stromal broad': 'Stromal broad',
#    'Fibroblasts + PVL': 'Stromal broad',
#    'B-cells': 'Immune broad',
#    'T-cells':'Immune broad',
#    'Myeloid': 'Immune broad',
#    'Immune broad': 'Immune broad',               
#    'Epithelial broadl': 'Epithelial broad',
#    'None': 'None'}


adata_sc = sc.read_h5ad("/Users/aslihankullelioglu/Downloads/txsim/sc_normalized.h5ad")
adata_sp_baysor_list = glob.glob('/Users/aslihankullelioglu/Downloads/txsim/spatial/*.h5ad')
df_list = []
for adata_sp_name in adata_sp_baysor_list[:5]: #first 5 to test
    # adata_sp.obs = adata_sp.obs.rename(columns={'celltype': 'celltype_panel'})
    # adata_sp.obs['celltype'] = [(panel_to_celltype)[k] for k in adata_sp.obs['celltype_panel']]
    # adata_sp.obs['celltype'] = adata_sp.obs['celltype'].astype('category')

    adata_sp = sc.read_h5ad(adata_sp_name)
    if re.match('baysor', adata_sp_name):
        adata_sp = adata_sp[~np.isnan(adata_sp.layers['lognorm']).any(axis=1)]
        adata_sp.write_h5ad("new_"+os.path.basename(adata_sp_name))
        
    
    output = all_metrics(adata_sp,adata_sc)
    df_list.append(output)

#spatial_data_path = "/Users/aslihankullelioglu/Downloads/txsim/spatial"
#spatial_files = [f for f in listdir(spatial_data_path) if isfile(join(spatial_data_path,f))]
# for file in adata_sp_baysor_list[:5]:
#     if file != '.DS_Store':
#         curr_whole_path = spatial_data_path + '/' + file
#         curr_adata_sp = sc.read_h5ad(curr_whole_path)
#         sc.pp.filter_cells(curr_adata_sp, min_genes=1, inplace=True)
#         curr_adata_sp.obs = curr_adata_sp.obs.rename(columns={'celltype': 'celltype_panel'})
#         curr_adata_sp.obs['celltype'] = [(panel_to_celltype)[k] for k in curr_adata_sp.obs['celltype_panel']]
#         curr_adata_sp.obs['celltype'] = curr_adata_sp.obs['celltype'].astype('category')
#         output = all_metrics(curr_adata_sp,adata_sc)
#         df_list.append(output)





# whole_df = pd.concat(df_list,axis=1)
# whole_df.columns = spatial_files[:5]
# #whole_df.fillna(value=0, inplace=True)
# whole_df = whole_df.T
# print(whole_df.dtypes)
# #sns.heatmap(whole_df, annot=True)


