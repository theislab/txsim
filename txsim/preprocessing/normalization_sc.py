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
    adata.layers['raw'] = adata.X
    adata.layers['norm'] = sc.pp.normalize_total(adata=adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed, 
        max_fraction=max_fraction, key_added=key_added, layer=layer, copy=copy, inplace=False)['X']
    adata.layers['lognorm'] = adata.layers['norm'].copy()
    sc.pp.log1p(adata, layer='lognorm')
    adata.X = adata.layers['lognorm']

    return adata
