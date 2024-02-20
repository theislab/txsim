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
    if measure_center not in ["mean", "median", "Mean", "Median", "MEAN", "MEDIAN"]:
        raise Exception (f"please chose either mean/Mean/MEAN or median/Median/MEDIAN instead of {measure_center}")
    
    adata_sc = adata_sc[:,adata_sp.var_names]
    if measure_center in ["median", "Median", "MEDIAN"]:
         median_reads_sp=np.median(np.sum(adata_sp.layers['raw'],axis=1), axis=0)
         median_reads_sp = float(median_reads_sp.flatten())
         median_reads_sc=np.median(np.sum(adata_sc.layers['raw'],axis=1), axis= 0)
         median_reads_sc = float(median_reads_sc.flatten())
         ratio_median_reads_cell = round(median_reads_sp/median_reads_sc, 3)
         return ratio_median_reads_cell
    
    elif measure_center in["mean", "Mean", "MEAN"]:
        mean_reads_sp=np.mean(np.sum(adata_sp.layers['raw'],axis=1), axis= 0)
        mean_reads_sp = float(mean_reads_sp.flatten())
        mean_reads_sc=np.mean(np.sum(adata_sc.layers['raw'],axis=1), axis= 0)
        mean_reads_sc = float(mean_reads_sc.flatten())
        ratio_mean_reads_cell= round(mean_reads_sp/mean_reads_sc, 3)
        return ratio_mean_reads_cell


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
    if measure_center not in ["mean", "median", "Mean", "Median", "MEAN", "MEDIAN"]:
        raise Exception (f"please chose either mean/Mean/MEAN or median/Median/MEDIAN instead of {measure_center}")
    
    adata_sc = adata_sc[:,adata_sp.var_names]
    if measure_center in ["median", "Median", "MEDIAN"]:
        median_genes_sp=np.median(np.sum((adata_sp.layers['raw']>0)*1,axis=1), axis=0)
        median_genes_sp = float(median_genes_sp.flatten())
        median_genes_sc=np.median(np.sum((adata_sc.layers['raw']>0)*1,axis=1), axis=0)
        median_genes_sc = float(median_genes_sc.flatten())
        ratio_median_genes_cell = round(median_genes_sp/median_genes_sc, 3)
        return ratio_median_genes_cell

    elif measure_center in ["mean", "Mean", "MEAN"]:
        mean_genes_sp=np.mean(np.sum((adata_sp.layers['raw']>0)*1,axis=1), axis= 0)
        mean_genes_sp = float(mean_genes_sp.flatten())
        mean_genes_sc=np.mean(np.sum((adata_sc.layers['raw']>0)*1,axis=1), axis= 0)
        mean_genes_sc = float(mean_genes_sc.flatten())
        ratio_mean_genes_cell = round(mean_genes_sp/mean_genes_sc, 3)
        return ratio_mean_genes_cell






    
    


    

