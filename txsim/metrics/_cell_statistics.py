import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData

def calculate_ratio_reads_per_cell(adata_sp: AnnData, adata_sc: AnnData, measure_center: str, pipeline_output = True):
    """Ratio between the median and mean number of reads/cells between spatial and  scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from single-cell data
    measure-center: str
        choice whether mean or median number of reads per cell is used to calculate ratio
    pipeline_output = True: Boolean whether to use the returns
    Returns
    -------
    median_reads : float
        median number of reads/cells between spatial and  scRNAseq
    mean_reads : float
        median number of reads/cells between spatial and  scRNAseq
    """
    adata_sc = adata_sc[:,adata_sp.var_names]
    if measure_center == "median":
         median_reads_sp=np.median(np.sum(adata_sp.layers['raw'],axis=1))
         median_reads_sc=np.median(np.sum(adata_sc.layers['raw'],axis=1))
         ratio_median_reads_cell = median_reads_sp/median_reads_sc
         return ratio_median_reads_cell
    
    if measure_center == "mean":
        mean_reads_sp=np.mean(np.sum(adata_sp.layers['raw'],axis=1))
        mean_reads_sc=np.mean(np.sum(adata_sc.layers['raw'],axis=1))
        ratio_mean_reads_cell=mean_reads_sp/mean_reads_sc
        return ratio_mean_reads_cell
    else:
        return f"please chose either mean or median instead of {measure_center}"


def calculate_ratio_genes_per_cell(adata_sp: AnnData,adata_sc: AnnData, measure_center: str, pipeline_output = True):
    """Ratio between the median and mean number of genes/cells between spatial and  scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from single-cell data
     measure-center: str
        choice whether mean or median number of reads per cell is used to calculate ratio
    pipeline_output = True: Boolean whether to use the returns
    Returns
    -------
    median_genes : float
        median number of reads/cells between spatial and  scRNAseq
    mean_genes : float
        median number of reads/cells between spatial and  scRNAseq
    """   
    adata_sc = adata_sc[:,adata_sp.var_names]
    if measure_center == "median":
        median_genes_sp=np.median(np.sum((adata_sp.layers['raw']>0)*1,axis=1))
        median_genes_sc=np.median(np.sum((adata_sc.layers['raw']>0)*1,axis=1))
        ratio_median_genes_cell = median_genes_sp/median_genes_sc
        return ratio_median_genes_cell

    if measure_center == "mean":
        mean_genes_sp=np.mean(np.sum((adata_sp.layers['raw']>0)*1,axis=1))
        mean_genes_sc=np.mean(np.sum((adata_sc.layers['raw']>0)*1,axis=1))
        ratio_mean_genes_cell = mean_genes_sp/mean_genes_sc
        return ratio_mean_genes_cell
    
    else:
        return f"please chose either mean or median instead of {measure_center}"






    
    


    

