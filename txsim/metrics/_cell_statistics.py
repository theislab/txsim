import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
import anndata

def ratio_median_readsXcells(adata_sp: AnnData,adata_sc: AnnData,pipeline_output=True):
    """Ratio between the median number of reads/cells between spatial and  scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
        median number of reads/cells between spatial and  scRNAseq
    """   
    adata_sc = adata_sc[:,adata_sp.var_names]
    median_cells_sp=np.median(np.sum(adata_sp.layers['raw'],axis=1))
    median_cells_sc=np.median(np.sum(adata_sc.layers['raw'],axis=1))
    ratio_median_reads_cell=median_cells_sp/median_cells_sc
    return ratio_median_reads_cell

def ratio_mean_readsXcells(adata_sp: AnnData,adata_sc: AnnData,pipeline_output=True):
    """Ratio between the mean number of reads/cells between spatial and  scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
        ration between the mean number of reads/cells between spatial and  scRNAseq
    """   
    adata_sc = adata_sc[:,adata_sp.var_names]
    mean_cells_sp=np.mean(np.sum(adata_sp.layers['raw'],axis=1))
    mean_cells_sc=np.mean(np.sum(adata_sc.layers['raw'],axis=1))
    ratio_mean_reads_cell=mean_cells_sp/mean_cells_sc
    return ratio_mean_reads_cell

def ratio_number_of_cells(adata_sp: AnnData,adata_sc: AnnData,pipeline_output=True):
    """ Ratio number of cells present in the spatial dataset vs scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    number_of cells : float
       Ratio between the number of cells present in the spatial dataset vs scRNAseq
    """   
    number_of_cells_scRNAseq=adata_sc.shape[0]
    number_of_cells_spatial=adata_sp.shape[0]
    ratio=number_of_cells_spatial/number_of_cells_scRNAseq
    return ratio

def ratio_mean_genesXcells(adata_sp: AnnData,adata_sc: AnnData,pipeline_output=True):
    """Ratio between the mean number of genes/cells between spatial and  scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
        ration between the mean number of genes/cells between spatial and  scRNAseq
    """   
    adata_sc = adata_sc[:,adata_sp.var_names]
    mean_cells_sp=np.mean(np.sum((adata_sp.layers['raw']>0)*1,axis=1))
    mean_cells_sc=np.mean(np.sum((adata_sc.layers['raw']>0)*1,axis=1))
    ratio_mean_genes_cell=mean_cells_sp/mean_cells_sc
    return ratio_mean_genes_cell

def ratio_median_genesXcells(adata_sp: AnnData,adata_sc: AnnData,pipeline_output=True):
    """Ratio between the median number of genes/cells between spatial and  scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
        ration between the median number of genes/cells between spatial and  scRNAseq
    """   
    adata_sc = adata_sc[:,adata_sp.var_names]
    median_cells_sp=np.median(np.sum((adata_sp.layers['raw']>0)*1,axis=1))
    median_cells_sc=np.median(np.sum((adata_sc.layers['raw']>0)*1,axis=1))
    ratio_median_genes_cell=median_cells_sp/median_cells_sc
    return ratio_median_genes_cell

