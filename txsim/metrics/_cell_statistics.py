import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad

def reads_per_cell_ratio(adata_sp: ad.AnnData, adata_sc: ad.AnnData, statistic: str = "mean", pipeline_output = True):
    """Ratio between the median or mean number of reads per cell between spatial and scRNAseq
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from single-cell data
    statistic: str
        Whether to calculate ratio of means or medians of reads per cell. Options: "mean" or "median"
    pipeline_output: bool
        Whether to return only a summary score or additionally also cell type level scores. TODO: implement per cell type scores
        
    Returns
    -------
    float:
        ratio of median or mean number of reads per cell between spatial and  scRNAseq
        
    """
    if statistic not in ["mean", "median"]:
        raise Exception (f"please choose either mean or median instead of {statistic}")
    
    adata_sc = adata_sc[:,adata_sp.var_names]
    if statistic == "median":
         median_reads_sp=np.median(np.sum(adata_sp.layers['raw'],axis=1), axis=0)
         median_reads_sp = float(median_reads_sp.flatten())
         median_reads_sc=np.median(np.sum(adata_sc.layers['raw'],axis=1), axis= 0)
         median_reads_sc = float(median_reads_sc.flatten())
         ratio_median_reads_cell = median_reads_sp/median_reads_sc
         return ratio_median_reads_cell
    elif statistic == "mean":
        mean_reads_sp=np.mean(np.sum(adata_sp.layers['raw'],axis=1), axis= 0)
        mean_reads_sp = float(mean_reads_sp.flatten())
        mean_reads_sc=np.mean(np.sum(adata_sc.layers['raw'],axis=1), axis= 0)
        mean_reads_sc = float(mean_reads_sc.flatten())
        ratio_mean_reads_cell= mean_reads_sp/mean_reads_sc
        return ratio_mean_reads_cell


def genes_per_cell_ratio(adata_sp: ad.AnnData,adata_sc: ad.AnnData, statistic: str = "mean", pipeline_output = True):
    """Ratio between the median or mean number of genes per cell between spatial and scRNAseq
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from single-cell data
    statistic: str
        Whether to calculate ratio of means or medians of reads per cell. Options: "mean" or "median"
    pipeline_output: bool
        Whether to return only a summary score or additionally also cell type level scores. TODO: implement per cell type scores
    
    Returns
    -------
    float:
        ratio of median or mean number of genes per cell between spatial and scRNAseq
    """
    if statistic not in ["mean", "median"]:
        raise Exception (f"please choose either mean or median instead of {statistic}")
    
    adata_sc = adata_sc[:,adata_sp.var_names]
    if statistic == "median":
        median_genes_sp=np.median(np.sum((adata_sp.layers['raw']>0)*1,axis=1), axis=0)
        median_genes_sp = float(median_genes_sp.flatten())
        median_genes_sc=np.median(np.sum((adata_sc.layers['raw']>0)*1,axis=1), axis=0)
        median_genes_sc = float(median_genes_sc.flatten())
        ratio_median_genes_cell = median_genes_sp/median_genes_sc
        return ratio_median_genes_cell

    elif statistic == "mean":
        mean_genes_sp=np.mean(np.sum((adata_sp.layers['raw']>0)*1,axis=1), axis= 0)
        mean_genes_sp = float(mean_genes_sp.flatten())
        mean_genes_sc=np.mean(np.sum((adata_sc.layers['raw']>0)*1,axis=1), axis= 0)
        mean_genes_sc = float(mean_genes_sc.flatten())
        ratio_mean_genes_cell = mean_genes_sp/mean_genes_sc
        return ratio_mean_genes_cell
