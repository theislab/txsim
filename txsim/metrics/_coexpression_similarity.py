from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc

#TODO Change how normalization happens and consider using log1p
def coexpression_similarity(
    spatial_data: AnnData,
    seq_data: AnnData,
    min_cells: int = 20,
    thresh: float = 0,
    layer: str = 'lognorm',
    key: str = 'celltype',
    by_celltype: bool = False,
    pipeline_output: bool = True,
):
    """Calculate the mean difference of normalised mutual information matrix values
    
    Parameters
    ----------
    spatial_data : AnnData
        annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    min_cells : int, optional
        Minimum number of cells in which a gene should be detected to be considered
        expressed. By default 20
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
    float
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

    print(len(common), "genes are shared in both modalities")
    
    
    if not by_celltype:
        	
        spatial_expressed = spt.var_names[sc.pp.filter_genes(spt, min_cells = min_cells, inplace=False)[0]]
        seq_expressed = seq.var_names[sc.pp.filter_genes(seq, min_cells = min_cells, inplace=False)[0]]
        
        print(round(len(spatial_expressed)/len(common) * 100), "% of genes are expressed in the spatial modality", sep = '')
        print(round(len(seq_expressed)/len(common) * 100), "% of genes are expressed in the single-cell modality", sep = '')
        
        common_exp = spatial_expressed.intersection(seq_expressed)
        
        if(not len(common_exp)):
            print("No expressed genes are shared in both modalities")
            output = None if pipeline_output else [None, None, None]
        else:
            print(len(common_exp), "out of", len(common), 'genes will be used')
            
            spt_exp = spt[:, common_exp]
            seq_exp = seq[:, common_exp]
        
            spt_mat = spt_exp.layers[layer].T
            seq_mat = seq_exp.layers[layer].T
            
            print("Calculating coexpression similarity")
            output = compute_mutual_information(spt_mat, seq_mat, common_exp, thresh, pipeline_output)
    else:
        output = {}
        # Determine cell type populations across both modalities
        ct_spt = list(np.unique(np.array(spt.obs[key])))
        ct_seq = list(np.unique(np.array(seq.obs[key])))
    
        common_types = [x for x in ct_spt if x in ct_seq]
        
        #spt_ct_counts = spt.obs[key].value_counts()
        #seq_ct_counts = seq.obs[key].value_counts()
        #print(type(spt_ct_counts[spt_ct_counts > min_cells]))
          
        for c in common_types:
            print("[%s]" % c)
        
            # Extract expression data layer
            spt_ct = spt[spt.obs[key]==c,:]
            seq_ct = seq[seq.obs[key]==c,:]
            
            # Extract genes expressed in at least 20 cells
            spatial_expressed = spt_ct.var_names[sc.pp.filter_genes(spt_ct, min_cells = min_cells, inplace=False)[0]]
            seq_expressed = seq_ct.var_names[sc.pp.filter_genes(seq_ct, min_cells = min_cells, inplace=False)[0]]
        
            print(round(len(spatial_expressed)/len(common) * 100), "% of genes are expressed in the spatial modality", sep = '')
            print(round(len(seq_expressed)/len(common) * 100), "% of genes are expressed in the single-cell modality", sep = '')
            
            common_exp = spatial_expressed.intersection(seq_expressed)

            if(not len(common_exp)):
                print("No expressed genes are shared in both modalities. Returning None value\n")
                output[c] = None
            else:
                print(len(common_exp), "out of", len(common), 'genes will be used')
            
                spt_exp = spt_ct[:, common_exp]
                seq_exp = seq_ct[:, common_exp]
        
                spt_mat = spt_exp.layers[layer].T
                seq_mat = seq_exp.layers[layer].T
 
                print("Calculating coexpression similarity")
                output[c] = compute_mutual_information(spt_mat, seq_mat, common_exp, thresh, pipeline_output)
            
       
    return(output)



def compute_mutual_information(spt_mat, seq_mat, common, thresh, pipeline_output):
    from pyitlib import discrete_random_variable as drv
    # Apply distance metric
    print("  - Spatial data...")        
    sim_spt = drv.information_mutual_normalised(spt_mat)
    print("  - Single-cell data...\n")
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
            print("The following genes in the single-cell modality resulted in NaN values:")
            for i in genes_nan: print(i)
                
        #If threshold, mask values with NaNs
        sim_seq[np.abs(sim_seq) < np.abs(thresh)] = np.nan
        sim_seq[np.tril_indices(len(common))] = np.nan


        # Calculate difference between modalities
        diff = sim_seq - sim_spt
        mean = np.nanmean(np.absolute(diff)) / 2
        output = mean
    
    return(output)
