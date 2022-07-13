import scanpy as sc
import anndata as ad
import numpy as np

#TODO replace with better name

def calc_metric1 (
    spatial_data,
    seq_data
):
    """
    :param spatial_data: annotated ``anndata`` object with counts from spatial data
    :param seq_data: annotated ``anndata`` object with counts scRNAseq data
    :returns: median value of difference between 
    """
    #Create matrix only with intersected genes
    common = seq_data.var_names.intersection(spatial_data.var_names)
    seq = seq_data[:, common]
    spt = spatial_data[:,common]
    #calculate corrcoef
    cor_seq = np.corrcoef(seq.X, rowvar=False)
    cor_spt = np.corrcoef(spt.X, rowvar=False)
    #subtract matricies
    diff = cor_seq - cor_spt
    #find median
    diff[np.tril_indices(len(common))] = np.nan
    mean = np.nanmean(np.absolute(diff)) / 2

    return mean
