import numpy as np
from anndata import AnnData
import scanpy as sc
from typing import Optional, Dict

def normalize_sc(
    adata: AnnData,
    target_sum: Optional[float] = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: Optional[str] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Scanpy normalize_total wrapper function
    Based on https://github.com/scverse/scanpy version 1.9.1

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to genes.
    target_sum : Optional[float], optional
        If `None`, after normalization, each observation (cell) has a total
        count equal to the median of total counts for observations (cells)
        before normalization, by default None
    exclude_highly_expressed : bool, optional
         Exclude (very) highly expressed genes for the computation of the
        normalization factor (size factor) for each cell. A gene is considered
        highly expressed, if it has more than `max_fraction` of the total counts
        in at least one cell. The not-excluded genes will sum up to
        `target_sum`, by default False
    max_fraction : float, optional
        If `exclude_highly_expressed=True`, consider cells as highly expressed
        that have more counts than `max_fraction` of the original total counts
        in at least one cell, by default 0.05
    key_added : Optional[str], optional
        Name of the field in `adata.obs` where the normalization factor is
        stored, by default None
    layer : Optional[str], optional
        Layer to normalize instead of `X`. If `None`, `X` is normalized, by default None
    inplace : bool, optional
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`, by default True
    copy : bool, optional
        Whether to modify copied input object. Not compatible with ``inplace=False``, by default False

    Returns
    -------
    Optional[Dict[str, np.ndarray]]
        Returns dictionary with normalized copies of `adata.X` and `adata.layers`
        or updates `adata` with normalized version of the original
    `   adata.X` and `adata.layers`, depending on `inplace`
    """
    if layer:
        adata.layers['raw'] = adata.layers[layer]
    else:
        adata.layers['raw'] = adata.X
    adata.layers['norm'] = sc.pp.normalize_total(adata=adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed, 
        max_fraction=max_fraction, key_added=key_added, layer=layer, copy=copy, inplace=False)['X']
    adata.layers['lognorm'] = adata.layers['norm'].copy()
    sc.pp.log1p(adata, layer='lognorm')
    adata.X = adata.layers['lognorm']

    return adata



# def gene_efficiency_correction(
#     adata_sp: AnnData,adata_sc: AnnData, layer_key:str='lognorm') -> AnnData:
#     """
#     Calculate the efficiency of every gene in the panel and normalize for that in the spatial object
#     Based on https://github.com/scverse/scanpy version 1.9.1

#     Parameters
#     ----------
#     adata : AnnData
#         The annotated data matrix of shape `n_obs` x `n_vars`.
#         Rows correspond to cells and columns to genes.
#     target_sum : Optional[float], optional
#         If `None`, after normalization, each observation (cell) has a total
#         count equal to the median of total counts for observations (cells)
#         before normalization, by default None
#     exclude_highly_expressed : bool, optional
#          Exclude (very) highly expressed genes for the computation of the
#         normalization factor (size factor) for each cell. A gene is considered
#         highly expressed, if it has more than `max_fraction` of the total counts
#         in at least one cell. The not-excluded genes will sum up to
#         `target_sum`, by default False
#     max_fraction : float, optional
#         If `exclude_highly_expressed=True`, consider cells as highly expressed
#         that have more counts than `max_fraction` of the original total counts
#         in at least one cell, by default 0.05
#     key_added : Optional[str], optional
#         Name of the field in `adata.obs` where the normalization factor is
#         stored, by default None
#     layer : Optional[str], optional
#         Layer to normalize instead of `X`. If `None`, `X` is normalized, by default None
#     inplace : bool, optional
#         Whether to update `adata` or return dictionary with normalized copies of
#         `adata.X` and `adata.layers`, by default True
#     copy : bool, optional
#         Whether to modify copied input object. Not compatible with ``inplace=False``, by default False

#     Returns
#     -------
#     adata_sp:AnnData
#         Adata object with the modified layer inputed in the key_layer
#     """

#     transform=layer_key
#     key='celltype'
#     adata_sc=adata_sc[:,adata_sp.var_names]
#     unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
#     genes=adata_sc.var.index[adata_sc.var.index.isin(adata_sp.var.index)]
#     exp_sc=pd.DataFrame(adata_sc.layers[transform],columns=adata_sc.var.index)
#     gene_means_sc=pd.DataFrame(np.mean(exp_sc,axis=0))
#     gene_means_sc=gene_means_sc.loc[gene_means_sc.index.sort_values(),:]
#     exp_sp=pd.DataFrame(adata_sp.layers[transform],columns=adata_sp.var.index)
#     gene_means_sp=pd.DataFrame(np.mean(exp_sp,axis=0))
#     gene_means_sp=gene_means_sp.loc[gene_means_sp.index.sort_values(),:]
#     exp_sc['celltype']=list(adata_sc.obs['celltype'])
#     exp_sp['celltype']=list(adata_sp.obs['celltype'])
#     exp_sc=exp_sc.loc[exp_sc['celltype'].isin(unique_celltypes),:]
#     exp_sp=exp_sp.loc[exp_sp['celltype'].isin(unique_celltypes),:]
#     mean_celltype_sp=exp_sp.groupby('celltype').mean()
#     mean_celltype_sc=exp_sc.groupby('celltype').mean()
#     mean_celltype_sc=mean_celltype_sc.loc[:,mean_celltype_sc.columns.sort_values()]
#     mean_celltype_sp=mean_celltype_sp.loc[:,mean_celltype_sp.columns.sort_values()]
#     #If no read is prestent in a gene, we add 0.1 so that we can compute statistics
#     mean_celltype_sp.loc[:,list(mean_celltype_sp.sum(axis=0)==0)]=0.001
#     mean_celltype_sc.loc[:,list(mean_celltype_sc.sum(axis=0)==0)]=0.001
#     gene_ratios=pd.DataFrame(np.mean(mean_celltype_sp,axis=0)/np.mean(mean_celltype_sc,axis=0))
#     gr=pd.DataFrame(gene_ratios)
#     gr.columns=['efficiency_st_vs_sc']
#     efficiency_mean=np.mean(gene_ratios)
#     efficiency_std=np.std(gene_ratios)
#     meanexp=pd.DataFrame(adata_sp.layers[transform],columns=adata_sp.var.index)
#     for gene in meanexp.columns:
#         meanexp.loc[:,gene]=meanexp.loc[:,gene]/gene_ratios.loc[gene,'efficiency_st_vs_sc']
#     adata_sp.layers[transform]=meanexp
#     return adata_sp