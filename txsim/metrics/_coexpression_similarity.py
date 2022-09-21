from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from pyitlib import discrete_random_variable as drv


def coexpression_similarity(
    spatial_data: AnnData,
    seq_data: AnnData,
    thresh: float = 0,
    return_matrices: bool = False
):
    """Calculate the mean difference of normalised mutual information matrix values
    
    Parameters
    ----------
    spatial_data : AnnData
        annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    thresh : float, optional
        threshold for significant pairs from scRNAseq data. Pairs with correlations
        below the threshold (by magnitude) will be ignored when calculating mean, by
        default 0
    return_matrices: bool
        return coexpression similarity matrix for each modality?
        default False
        
    Returns
    -------
    mean : float
        mean of upper triangular difference matrix
    matrices: list
        list containing coexpression similarity matrix for each modality and gene names
    """

    #Copy to prevent overwriting original
    _seq_data = seq_data.copy()
    _spatial_data = spatial_data.copy()


    _spatial_data.X = _spatial_data.layers['lognorm']
    _seq_data.X = _seq_data.layers['lognorm']

    #Create matrix only with intersected genes
    common = _seq_data.var_names.intersection(_spatial_data.var_names)
    seq = _seq_data[:, common]
    spt = _spatial_data[:,common]

    # Apply distance metric
    print("Calculating coexpression for:")
    print("  - Spatial data...")
    sim_spt = drv.information_mutual_normalised(spt.X.T)
    print("  - Single-cell data...")
    sim_seq = drv.information_mutual_normalised(seq.X.T)
    
    
    if return_matrices:
        return([sim_spt, sim_seq, common])
    else:
        # Evaluate NaN values for each gene in every modality

        ## Spatial
        nan_res = np.sum(np.isnan(sim_spt), axis = 0)
        if any(nan_res == len(common)):
            genes_nan = common[nan_res == len(common)]
            genes_nan = genes_nan.tolist()
            print("The following genes in the spatial modality resulted in NaN values:")
            for i in genes_nan: print(i)

        ## Single cell
        nan_res = np.sum(np.isnan(sim_seq), axis = 0)
        if any(nan_res == len(common)):
            genes_nan = common[nan_res == len(common)]
            genes_nan = genes_nan.tolist()
            print("The following genes in the single-cell modality resulted in NaN values")
            for i in genes_nan: print(i)


        #If threshold, mask values with NaNs
        sim_seq[np.abs(sim_seq) < np.abs(thresh)] = np.nan
        sim_seq[np.tril_indices(len(common))] = np.nan


        # Calculate difference between modalities
        diff = sim_seq - sim_spt
        mean = np.nanmean(np.absolute(diff)) / 2
        return mean


def coexpression_similarity_celltype(
    spatial_data: AnnData,
    seq_data: AnnData,
    thresh: float = 0,
    celltype: str = 'celltype'
) -> float:
    """Calculate the mean difference of Pearson correlation matrix values
    
    Parameters
    ----------
    spatial_data : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    thresh : float, optional
        Threshold for significant pairs from scRNAseq data. Pairs with correlations
        below the threshold (by magnitude) will be ignored when calculating mean, by
        default 0
    celltype : str, optional
        Key where cell types are stored in `adata.obs`, by default 'celltype'

    Returns
    -------
    mean : float
        Mean of upper triangular difference matrix
    """
    #Create matrix only with intersected genes
    common = seq_data.var_names.intersection(spatial_data.var_names)
    seq = seq_data[:, common]
    spt = spatial_data[:,common]
    seq.X = seq.layers['lognorm']
    spt.X = spt.layers['lognorm']
    common_types = set(seq.obs[celltype]).intersection(set(spt.obs[celltype]))

    mean_dict = {}
    for c in common_types:
        #If there is only 1 cell:, skip cell type
        if len(spt[spt.obs[celltype]==c]) == 1: continue

        #Calculate corrcoef
        cor_seq = np.corrcoef(seq[seq.obs[celltype]==c,:].X, rowvar=False)
        cor_spt = np.corrcoef(spt[spt.obs[celltype]==c,:].X, rowvar=False)

        proportion = len(spt[spt.obs[celltype]==c])/len(spt.obs)
        sc_proportion = len(seq[seq.obs[celltype]==c])/len(seq.obs)

        #If above threshold
        cor_seq[np.abs(cor_seq) < np.abs(thresh)] = np.nan
        cor_seq[np.tril_indices(len(common))] = np.nan
        
        #Find spatial correlations above threshold
        spt_above = cor_spt[np.abs(cor_spt) > np.abs(thresh)]
        spt_above = spt_above[spt_above < 0.9999]

        #Subtract matricies
        diff = cor_seq - cor_spt

        #Find mean of upper triangular
        mean = np.nanmean(np.absolute(diff)) / 2
        mean_dict[c] = [mean, len(spt_above), len(cor_seq[~np.isnan(cor_seq)]), proportion, sc_proportion]

    return pd.DataFrame.from_dict(mean_dict, orient='index', 
            columns=['mean_diff', ' spt_above', 'seq_above', 'pct', 'sc_pct'])
