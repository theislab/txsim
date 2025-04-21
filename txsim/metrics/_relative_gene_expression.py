import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from scipy import stats

def relative_gene_expression(
    adata_sp: AnnData,
    adata_sc: AnnData, 
    cell_type_key:str='cell_type',
    layer:str='normalized',  
    min_cells:int = 10,  
) -> float:
    """Calculate the efficiency deviation present between the genes in the panel. 
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    cell_type_key: str (default: 'cell_type')
        .obs column of ``AnnData`` that contains celltype information
    layer: str (default: 'normalized')
        layer of ```AnnData`` to use to compute the metric
    min_cells: int (default: 10)
        minimum number of cells needed for a cell type to be included

    Returns
    -------
    average_corr: float
        Spearman correlation of gene expression for same cell type in scRNA-seq vs spatial data. Simple average across all cell types
  
    """   
    from scipy import stats

    common_cell_types = [x for x in adata_sp.obs['cell_type'].unique() if x in adata_sc.obs['cell_type'].unique()] 
    spearman_corr = pd.Series(data=0, index = common_cell_types, dtype = np.float64)

    # for each cell type, calculate spearman or NaN if below min_cells
    for cell_type in common_cell_types:

        spatial_cells = adata_sp[adata_sp.obs['cell_type'] == cell_type]
        scrnaseq_cells = adata_sc[adata_sc.obs['cell_type'] == cell_type]
        
        #skip cell types with fewer than min_cells
        if len(spatial_cells.obs.index) < min_cells or len(scrnaseq_cells.obs.index) < min_cells: 
            spearman_corr[cell_type] = np.nan
            continue 

        avg_gene_expression_ref = np.average(scrnaseq_cells.layers['normalized'].toarray(), axis=0) #TODO can we assume normalized layer will be sparse?
        avg_gene_expression_spatial = np.average(spatial_cells.layers['normalized'].toarray(), axis=0)

        spearman_corr[cell_type] = stats.spearmanr(avg_gene_expression_ref, avg_gene_expression_spatial).statistic

    #return mean of spearman, ignoring NaN's
    return np.nanmean(spearman_corr)
