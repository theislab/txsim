import numpy as np
from anndata import AnnData
import scanpy as sc
from typing import Optional, Dict

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
    
    return sc.pp.normalize_total(adata=adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed, 
        max_fraction=max_fraction, key_added=key_added, layer=layer, inplace=inplace, copy=copy)

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

    return sc.experimental.pp.normalize_pearson_residuals(adata, theta=theta, clip=clip, check_values=check_values, 
        layer=layer, inplace=inplace, copy=copy)


#TODO Fill in function
def normalize_by_area(
    adata: AnnData,
    area: Optional[str] = 'area',
    inplace: Optional[bool] = True
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
    inplace : Optional[bool], optional
        If ``True``, update ``adata`` with results. Otherwise, return results, by default True

    Returns
    -------
    np.ndarray
        If ``inplace=True``, ``adata.X`` is updated with the normalized values. 
        Otherwise, returns normalized numpy array
    """
    x = adata.X / adata.obs[area][:,None]
    
    if(not inplace):
        return x
    adata.X = x
    return 
