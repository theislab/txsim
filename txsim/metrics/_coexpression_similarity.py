from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from pyitlib import discrete_random_variable as drv


def coexpression_similarity(
    spatial_data: AnnData,
    seq_data: AnnData,
    thresh: float=0,
    layer: str='lognorm',
    key: str='celltype',
    by_celltype: bool=True,
    pipeline_output: bool = True,
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
    layer : str, optional
        name of layer used to calculate coexpression similarity. Should be the same
        in both AnnData objects
        default lognorm
    key : str
        name of the column containing the cell type information
    by_celltype: bool
        run analysis by cell type? If False, computation will be performed using the
        whole gene expression matrix
    pipeline_output: bool, optional (default: True)
        flag whether the metric is run as part of the pipeline or not. If not, then
        coexpression similarity matrices are returned for each modality. Otherwise,
        only the mean absolute difference of the upper triangle of the coexpression
        matrices is returned as a single score.
        
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

    #Create matrix only with intersected genes
    common = _seq_data.var_names.intersection(_spatial_data.var_names)
    seq = _seq_data[:, common]
    spt = _spatial_data[:,common]

    # Determine cell type populations across both modalities
    ct_spt = list(np.unique(np.array(spt.obs[key])))
    ct_seq = list(np.unique(np.array(seq.obs[key])))
    
    common_types = [x for x in ct_spt if x in ct_seq]
          
    print("Calculating coexpression similarity")
    
    if not by_celltype:
        spt_mat = spt.layers[layer].T
        seq_mat = seq.layers[layer].T
        output = compute_mutual_information(spt_mat, seq_mat, common, thresh, pipeline_output)

    else:
        output = {}
        for c in common_types:
            #If there is only 1 cell:, skip cell type
            #if len(spt[spt.obs['celltype']==c]) == 1: continue
            print("[%s]" % c)
        
            # Extract expression data layer
            spt_ct = spt[spt.obs[key]==c,:]
            spt_mat = spt_ct.layers[layer].T
            
            seq_ct = seq[seq.obs[key]==c,:]
            seq_mat = seq_ct.layers[layer].T
            output[c] = compute_mutual_information(spt_mat, seq_mat, common, thresh, pipeline_output)
            
       
    return(output)



def compute_mutual_information(spt_mat, seq_mat, common, thresh, pipeline_output):
    # Apply distance metric
    print("  - Spatial data...")        
    sim_spt = drv.information_mutual_normalised(spt_mat)
    print("  - Single-cell data...")
    sim_seq = drv.information_mutual_normalised(seq_mat)
    
    if not pipeline_output:
        output = [sim_spt, sim_seq, common]
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
        output = mean
    
    return(output)
        
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
