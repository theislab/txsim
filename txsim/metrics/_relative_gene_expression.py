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
    method:str =  'spearman',
    min_cells_expressing_gene:float = 0.01
) -> float:
    """Calculate the correlation between the gene expression in scRNAseq and spatial data, averaged across cell types
    ----------
    adata_sp : AnnData
        Annotated ```AnnData``` object with counts from spatial data
    adata_sc : AnnData
        Annotated ```AnnData``` object with counts from scRNAseq data
    cell_type_key: str (default: 'cell_type')
        Column of ```.obs``` in ```AnnData``` that contains celltype information
    layer: str (default: 'normalized')
        Layer of ```AnnData`` to use to compute the metric
    min_cells: int (default: 10)
        Minimum number of cells in a cell type needed for that cell type to be included
    method: str (default: 'spearman')
        Method use to calculate correlation, options are 'spearman', 'pearson', and 'kendall' (for Kendall's tau-b)
    min_cells_expressing_gene: float (default: 0.01)
        Minimum fraction of cells in that cell type expressing a gene needed for that gene to be included in correlation.
        By default, genes expressed in <1% of cells for that cell type will not be used to calculate correlation. 
        Pass in 0 to calculate correlation across all genes- note this may result in zero-inflated data. 

    Returns
    -------
    average_corr: float
        Correlation of gene expression for same cell type in scRNA-seq vs spatial data. Simple average across all cell types
  
    """   
    from scipy import stats

    common_cell_types = [x for x in adata_sp.obs['cell_type'].unique() if x in adata_sc.obs['cell_type'].unique()] 
    correlations = pd.Series(data=0, index = common_cell_types, dtype = np.float64)

    # for each cell type, calculate correlation or NaN if below min_cells
    for cell_type in common_cell_types:

        spatial_cells = adata_sp[adata_sp.obs['cell_type'] == cell_type]
        scrnaseq_cells = adata_sc[adata_sc.obs['cell_type'] == cell_type]
        
        #skip cell types with fewer than min_cells
        if len(spatial_cells.obs.index) < min_cells: # or len(scrnaseq_cells.obs.index) < min_cells: #TODO um idk here
            correlations[cell_type] = np.nan
            continue 

        avg_gene_expression_ref = np.average(scrnaseq_cells.layers['normalized'].toarray(), axis=0) #TODO can we assume normalized layer will be sparse?
        avg_gene_expression_spatial = np.average(spatial_cells.layers['normalized'].toarray(), axis=0) #otherwise 'toarray' is not needed

        #calculate fraction of cells expressing each gene
        cell_fraction = np.average(scrnaseq_cells.layers['normalized'].toarray()>0, axis=0)

        #filter out genes below threshold
        filter_idx = np.where(cell_fraction >= min_cells_expressing_gene)
        avg_gene_expression_ref = avg_gene_expression_ref[filter_idx]
        avg_gene_expression_spatial = avg_gene_expression_spatial[filter_idx]

        if ('spearman' in method):
            correlations[cell_type] = stats.spearmanr(avg_gene_expression_ref, avg_gene_expression_spatial).statistic
        elif ('kendall' in method):
            correlations[cell_type] = stats.kendalltau(avg_gene_expression_ref, avg_gene_expression_spatial).statistic
        elif ('pearson' in method): 
            correlations[cell_type] = stats.pearsonr(avg_gene_expression_ref, avg_gene_expression_spatial).statistic
        else:
            print("invalid method used") #TODO make this warning better

    #return mean of correlations, ignoring NaN's
    return np.nanmean(correlations)
