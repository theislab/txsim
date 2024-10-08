import numpy as np
from anndata import AnnData
import scanpy as sc
import pandas as pd
from typing import Optional, Dict
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

def normalize_total(
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
    adata.layers['raw'] = adata.X
    adata.layers['norm'] = sc.pp.normalize_total(adata=adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed, 
        inplace=False, max_fraction=max_fraction, key_added=key_added, layer=layer, copy=copy)['X']
    adata.layers['lognorm'] = adata.layers['norm'].copy()
    sc.pp.log1p(adata, layer='lognorm')

    return adata

def normalize_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """Scanpy normalize_pearson_residuals wrapper function
    Based on https://github.com/scverse/scanpy version 1.9.1

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to genes.
    theta : float, optional
        The negative binomial overdispersion parameter ``theta`` for Pearson residuals. 
        Higher values correspond to less overdispersion (``var = mean + mean^2/theta``), 
        and ``theta=np.Inf`` corresponds to a Poisson model, by default 100
    clip : Optional[float], optional
        Determines if and how residuals are clipped:
            If ``None``, residuals are clipped to the interval ``[-sqrt(n_obs), sqrt(n_obs)]``, 
                where n_obs is the number of cells in the dataset (default behavior).
            If any scalar ``c``, residuals are clipped to the interval ``[-c, c]``. 
            Set ``clip=np.Inf`` for no clipping
        by default None
    check_values : bool, optional
        If ``True``, checks if counts in selected layer are integers as expected by this function, 
        and return a warning if non-integers are found. Otherwise, proceed without checking. 
        Setting this to ``False`` can speed up code for large datasets, by default True
    layer : Optional[str], optional
        Layer to use as input instead of ``X``. If ``None``, ``X`` is used, by default None
    inplace : bool, optional
        If ``True``, update ``adata`` with results. Otherwise, return results, by default True
    copy : bool, optional
        If ``True``, the function runs on a copy of the input object and returns the modified copy. 
        Otherwise, the input object is modified direcly. Not compatible with ``inplace=False``, by default False

    Returns
    -------
    Optional[Dict[str, np.ndarray]]
        If ``inplace=True``, ``adata.X`` or the selected layer in ``adata.layers`` is updated with the 
        normalized values. ``adata.uns`` is updated with the following fields. If ``inplace=False``, the same 
        fields are returned as dictionary with the normalized values in ``results_dict['X']``:
        
        ``.uns['pearson_residuals_normalization']['theta']`` : The used value of the overdisperion parameter theta.
        ``.uns['pearson_residuals_normalization']['clip']``: The used value of the clipping parameter.
        ``.uns['pearson_residuals_normalization']['computed_on']``:The name of the layer on which the residuals were computed.
    """

    adata.layers['raw'] = adata.X.copy()
    adata.layers['norm'] = sc.experimental.pp.normalize_pearson_residuals(adata, theta=theta, clip=clip, check_values=check_values, 
        layer=layer, inplace=False, copy=copy)['X']
    adata.layers['lognorm'] = adata.layers['norm'].copy()
    sc.pp.log1p(adata, layer='lognorm')
    return adata

def normalize_by_area(
    adata: AnnData,
    area: Optional[str] = 'area'
) -> Optional[np.ndarray]:
    """Normalize counts by area of cells

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to genes.
    area : Optional[str], optional
        Name of the field in `adata.obs` where the area is
        stored, by default 'area'

    Returns
    -------
    np.ndarray
        If ``inplace=True``, ``adata.X`` is updated with the normalized values. 
        Otherwise, returns normalized numpy array
    """
    
    adata.layers['raw'] = adata.X.copy()
    
    if issparse(adata.X):
        areas = adata.obs[area].copy()
        areas.loc[areas.isnull()] = 1 #Don't normalize cells that don't have an area
        areas = areas.to_numpy()
        sparsefuncs.inplace_row_scale(adata.X, 1 / areas)
    else:
        np.divide(adata.X, adata.obs[area].to_numpy()[:,None], out=adata.X)
        
    adata.layers['norm'] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers['lognorm'] = adata.X
    
    return adata



def gene_efficiency_correction(
    adata_sp: AnnData,adata_sc: AnnData, layer_key:str='lognorm', ct_key:str='celltype') -> AnnData:
    """
    Calculate the efficiency of every gene in the panel and normalize for that in the spatial object
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
    ct_key : str, optional
        Name of the field in `adata.obs` where the cell types are stored, by default 'celltype'

    Returns
    -------
    Optional[Dict[str, np.ndarray]]
        Returns dictionary with normalized copies of `adata.X` and `adata.layers`
        or updates `adata` with normalized version of the original
    `   adata.X` and `adata.layers`, depending on `inplace`
    """

    transform=layer_key
    adata_sc=adata_sc[:,adata_sp.var_names].copy()
    
    if issparse(adata_sc.layers[transform]):
        adata_sc.layers[transform]=adata_sc.layers[transform].toarray().copy()
    if issparse(adata_sp.layers[transform]):
        adata_sp.layers[transform]=adata_sp.layers[transform].toarray().copy()
        
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[ct_key].isin(adata_sp.obs[ct_key]),ct_key].unique()
    genes=adata_sc.var.index[adata_sc.var.index.isin(adata_sp.var.index)]
    exp_sc=pd.DataFrame(adata_sc.layers[transform],columns=adata_sc.var.index)
    gene_means_sc=pd.DataFrame(np.mean(exp_sc,axis=0))
    gene_means_sc=gene_means_sc.loc[gene_means_sc.index.sort_values(),:]
    exp_sp=pd.DataFrame(adata_sp.layers[transform],columns=adata_sp.var.index)
    gene_means_sp=pd.DataFrame(np.mean(exp_sp,axis=0))
    gene_means_sp=gene_means_sp.loc[gene_means_sp.index.sort_values(),:]
    exp_sc[ct_key]=list(adata_sc.obs[ct_key])
    exp_sp[ct_key]=list(adata_sp.obs[ct_key])
    exp_sc=exp_sc.loc[exp_sc[ct_key].isin(unique_celltypes),:]
    exp_sp=exp_sp.loc[exp_sp[ct_key].isin(unique_celltypes),:]
    mean_celltype_sp=exp_sp.groupby(ct_key).mean().astype(np.float64)
    mean_celltype_sc=exp_sc.groupby(ct_key).mean().astype(np.float64)
    mean_celltype_sc=mean_celltype_sc.loc[:,mean_celltype_sc.columns.sort_values()]
    mean_celltype_sp=mean_celltype_sp.loc[:,mean_celltype_sp.columns.sort_values()]
    #If no read is prestent in a gene, we add 0.1 so that we can compute statistics
    mean_celltype_sp.loc[:,list(mean_celltype_sp.sum(axis=0)==0)]=0.001
    mean_celltype_sc.loc[:,list(mean_celltype_sc.sum(axis=0)==0)]=0.001
    #mean_celltype_sp_norm=mean_celltype_sp.div(mean_celltype_sp.mean(axis=0),axis=1)
    #mean_celltype_sc_norm=mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)
    gene_ratios=pd.DataFrame(np.mean(mean_celltype_sp,axis=0)/np.mean(mean_celltype_sc,axis=0))
    gr=pd.DataFrame(gene_ratios)
    gr.columns=['efficiency_st_vs_sc']
    efficiency_mean=np.mean(gene_ratios,axis=0)
    efficiency_std=np.std(gene_ratios,axis=0)
    meanexp=pd.DataFrame(adata_sp.layers[transform],columns=adata_sp.var.index)
    for gene in meanexp.columns:
        meanexp.loc[:,gene]=meanexp.loc[:,gene]/gr.loc[gene,'efficiency_st_vs_sc']
    adata_sp.layers[transform]=meanexp
    return adata_sp