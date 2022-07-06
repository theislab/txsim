import numpy as np
import anndata as ad
import scanpy as sc

def normalize_total(
    adata, 
    target_sum=None, 
    exclude_highly_expressed=False, 
    max_fraction=0.05, 
    key_added=None, 
    layer=None, 
    layers=None, 
    layer_norm=None, 
    inplace=True, 
    copy=False
):
    """Scanpy normalize_total wrapper function
    Based on https://github.com/scverse/scanpy version 1.9.1

    :param adata: unnormalized ``anndata`` object with counts
    :param target_sum: float target for normalizing total counts, if ``None`` total count will 
        equal median of counts before normalization
    :param exclude_highly_expressed: Exclude highly expressed genes (a gene is highly expressed if it has 
        more than ``max_fraction`` of total counts for at least one cell). Included genes will sum to ``target_sum``
    :param max_fraction: if ``exclude_highly_expressed=True``, consider genes as highly expressed if they have 
        more than ``max_fraction`` of total counts in at least one cell 
    :param key_added: name of field in ``adata.obs`` where normalization factor is stored
    :param layer: layer to normalize instead of ``X``. if ``None``, ``X`` is normalized  
    :param inplace: Whether to update adata or return dictionary with normalized copies 
        of ``adata.X`` and ``adata.layers``
    :param copy: Whether to modify copied input object. Not compatible with ``inplace=False``.
    :return:Returns dictionary with normalized copies of ``adata.X`` and ``adata.layers`` or 
        updates adata with normalized version of the original ``adata.X`` and ``adata.layers``, 
        depending on ``inplace``
    """
    return sc.pp.normalize_total(adata, target_sum, exclude_highly_expressed, 
        max_fraction, key_added, layer, layers, layer_norm, inplace, copy)

def normalize_pearson_residuals(
    adata, 
    *, 
    theta=100, 
    clip=None, 
    check_values=True, 
    layer=None, 
    inplace=True, 
    copy=False
):
    """Scanpy normalize_pearson_residuals wrapper function
    Based on https://github.com/scverse/scanpy version 1.9.1

    :param adata: annotated ``anndata`` object with counts
    :param theta: float for the negative binomial overdispersion parameter ``theta`` for Pearson residuals. 
        Higher values correspond to less overdispersion (``var = mean + mean^2/theta``), 
        and ``theta=np.Inf`` corresponds to a Poisson model.
    :param clip: Determines if and how residuals are clipped:
        If ``None``, residuals are clipped to the interval ``[-sqrt(n_obs), sqrt(n_obs)]``, 
            where n_obs is the number of cells in the dataset (default behavior).
        If any scalar ``c``, residuals are clipped to the interval ``[-c, c]``. 
        Set ``clip=np.Inf`` for no clipping.
    :param check_values: If ``True``, checks if counts in selected layer are integers as expected by this function, 
        and return a warning if non-integers are found. Otherwise, proceed without checking. 
        Setting this to ``False`` can speed up code for large datasets.
    :param layer: Layer to use as input instead of ``X``. If ``None``, ``X`` is used.
    :param inplace: If ``True``, update ``adata`` with results. Otherwise, return results.
    :param copy: If ``True``, the function runs on a copy of the input object and returns the modified copy. 
        Otherwise, the input object is modified direcly. Not compatible with ``inplace=False``.
    :return: If ``inplace=True``, ``adata.X`` or the selected layer in ``adata.layers`` is updated with the 
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
    adata
):
    return adata
    