from anndata import AnnData
import numpy as np

#TODO replace with better name

def coexpression_similarity(
    spatial_data: AnnData,
    seq_data: AnnData
) -> float:
    """Calculate the mean difference of Pearson correlation matrix values
    
    Parameters
    ----------
    spatial_data : AnnData
        annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        annotated ``AnnData`` object with counts scRNAseq data

    Returns
    -------
    mean : float
        mean of upper triangular difference matrix
    """
    #Create matrix only with intersected genes
    common = seq_data.var_names.intersection(spatial_data.var_names)
    seq = seq_data[:, common]
    spt = spatial_data[:,common]
    #Calculate corrcoef
    cor_seq = np.corrcoef(seq.X, rowvar=False)
    cor_spt = np.corrcoef(spt.X, rowvar=False)
    #Subtract matricies
    diff = cor_seq - cor_spt
    #Find mean of upper triangular
    diff[np.tril_indices(len(common))] = np.nan
    mean = np.nanmean(np.absolute(diff)) / 2

    return mean
