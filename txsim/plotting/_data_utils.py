
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from typing import Optional


def negative_marker_dotplot(
        adata_sc: AnnData, 
        adata_sp: Optional[AnnData] = None, 
        celltypes: Optional[list] = None, 
        genes: Optional[list] = None,
        sc_dotplot_kws: dict = {},
        save: Optional[str] = None,
    ):
    """
    
    Two options: either provide adata_sp or celltypes & genes. Useful to have both of them because there are cases
    where you don't have the adata_sp ready. Note that there is one difference: if an adata_sp was provided we also
    take into account the number of cells filter for shared celltypes selection.
    
    TODO: support adata_sp
    
    celltypes: list
        List of celltypes in spatial and single cell data. (None if adata_sp is given)
    genes: list
        List of genes in spatial and single cell data. (None if adata_sp is given)
    sc_dotplot_kws: dict
        Keyword arguments for scanpy.pl.dotplot.
    
    """
    
    if adata_sp is not None:
        raise NotImplementedError("adata_sp currently not supported, provide celltypes and genes instead.")
    
    key = "celltype"
    
    assert np.all([g in adata_sc.var_names for g in genes]); "Some gene in `genes` is not in `adata_sc.var_names`"
    celltypes_sc = adata_sc.obs[key].unique()
    assert np.all([c in celltypes_sc for c in celltypes]); "Some ct in `celltypes` is not in `adata_sc.obs['celltype']`"
    
    # Get negative markers-ct pairs (currently copied from NMP metric fct)
    #TODO: Exchange with the negative marker selection function tx.metrics.get_negative_marker_ct_pairs (needs to be written first)
    min_number_cells=10 # minimum number of cells belonging to a cluster to consider it in the analysis
    max_ratio_cells=0.005 # maximum ratio of cells expressing a marker to call it a negative marker gene-ct pair
    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,genes]
    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    if issparse(adata_sc.layers["raw"]):
        adata_sc.layers["raw"] = adata_sc.layers["raw"].toarray()
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[celltypes]
    ct_filter = (celltype_count_sc >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()
    # Return nan if too few cell types were found
    if len(celltypes) < 2:
        print("Not enough cell types (>1) eligible to calculate negative marker purity")
        negative_marker_purity = 'nan'
        return
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    # Get mean expression per cell type
    exp_sc = pd.DataFrame(adata_sc.layers['raw'],columns=genes)
    exp_sc['celltype'] = list(adata_sc.obs[key])
    mean_celltype_sc = exp_sc.groupby('celltype').mean()
    # Get mean expressions relative to mean over cell types (this will be used for filtering based on minimum_exp)
    mean_ct_sc_rel = mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)
    # Get normalized mean expressions over cell types (this will be summed up over negative cell types)
    mean_ct_sc_norm = mean_celltype_sc.div(mean_celltype_sc.sum(axis=0),axis=1)
    
    
    # Plot
    dp = sc.pl.dotplot(adata_sc,adata_sc.var_names,"celltype", show=False, **sc_dotplot_kws)
    plt.sca(dp['mainplot_ax'])
    plt.imshow(
        (mean_ct_sc_norm.loc[adata_sc.obs[key].cat.categories,adata_sc.var_names[::-1]] < max_ratio_cells), #0.005
        alpha=0.8, zorder=0, cmap="PuOr",
        extent=(len(adata_sc.var_names),0,len(mean_ct_sc_norm),0)
    )
    if save is not None:
        plt.gcf().savefig(save, bbox_inches = "tight", transparent=True)
    plt.show()

    #TODO: Add legend indicating colors for negative marker cell type pairs