import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from scipy.sparse import isspmatrix 
import itertools

# TODO: probably better to combine both metrics to one function with argument `flavor in ["reads", "cells"]`
# TODO: Write test
# TODO: Investigate the importance of setting max_ratio_cells, minimum_exp, and min_number_cells

#helper function 
def get_eligible_celltypes(adata_sp: AnnData, adata_sc: AnnData, key: str='celltype'):
    """ Get shared celltypes of adata_sp and adata_sc, that have at least min_number_cells members.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data

    Returns
    -------
    celltypes, adata_sp, adata_sc

    """
    # Set threshold parameters
    min_number_cells=10 # minimum number of cells belonging to a cluster to consider it in the analysis
    
    #check that genes in spatial data is subset of genes in single cell data
    adata_sp = adata_sp[:,adata_sp.var_names.isin(adata_sc.var_names)]

    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,adata_sp.var_names]
    
    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    for a in [adata_sc, adata_sp]:
        if issparse(a.layers["raw"]):
            a.layers["raw"] = a.layers["raw"].toarray()
    
    # Get cell types that we find in both modalities
    shared_celltypes = adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[shared_celltypes]
    celltype_count_sp = adata_sp.obs[key].value_counts().loc[shared_celltypes]      
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()

    return celltypes, adata_sp, adata_sc


def negative_marker_purity_cells(adata_sp: AnnData, adata_sc: AnnData, key: str='celltype', pipeline_output: bool=True):
    """ Negative marker purity aims to measure read leakeage between cells in spatial datasets. 
    
    For this, we calculate the increase in positive cells assigned in spatial datasets to pairs of genes-celltyes with 
    no/very low expression in scRNAseq.
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    negative marker purity : float
       Increase in proportion of positive cells assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    """ 
    max_ratio_cells=0.005 # maximum ratio of cells expressing a marker to call it a negative marker gene-ct pair  
    
    celltypes, adata_sp, adata_sc  = get_eligible_celltypes(adata_sp, adata_sc, key)
    
    # Return nan if too few cell types were found
    if len(celltypes) < 2:
        print("Not enough cell types (>1) eligible to calculate negative marker purity")
        negative_marker_purity = 'nan'
        if pipeline_output==True:
            return negative_marker_purity
        else:
            return negative_marker_purity, None, None
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp = adata_sp[adata_sp.obs[key].isin(celltypes)]
    
    # besides the threshold parameter till here neg. marker purity reads and cells are the same
    
    
    # Get ratio of positive cells per cell type
    pos_exp_sc = pd.DataFrame(adata_sc.layers['raw'] > 0,columns=adata_sp.var_names)
    pos_exp_sp = pd.DataFrame(adata_sp.layers['raw'] > 0,columns=adata_sp.var_names)
    pos_exp_sc['celltype'] = list(adata_sc.obs[key])
    pos_exp_sp['celltype'] = list(adata_sp.obs[key])
    ratio_celltype_sc = pos_exp_sc.groupby('celltype').mean()
    ratio_celltype_sp = pos_exp_sp.groupby('celltype').mean()
    
    # Get gene-cell type pairs with negative marker expression
    neg_marker_mask = np.array(ratio_celltype_sc < max_ratio_cells)
    
    # Return nan if no negative markers were found
    if np.sum(neg_marker_mask) < 1:
        print("No negative markers were found in the sc data reference.")
        negative_marker_purity = 'nan'
        if pipeline_output==True:
            return negative_marker_purity
        else:
            return negative_marker_purity, None, None
    
    # Get pos cell ratios in negative marker-cell type pairs
    lowvals_sc = np.full_like(neg_marker_mask, np.nan, dtype=np.float32)    
    lowvals_sc = ratio_celltype_sc.values[neg_marker_mask]
    lowvals_sp = ratio_celltype_sp.values[neg_marker_mask]
    
    # Take the mean over the normalized expressions of the genes' negative cell types
    mean_sc_low_ratio = np.nanmean(lowvals_sc)
    mean_sp_low_ratio = np.nanmean(lowvals_sp)
    
    # Calculate summary metric
    negative_marker_purity = 1
    if mean_sp_low_ratio > mean_sc_low_ratio:
        negative_marker_purity -= (mean_sp_low_ratio - mean_sc_low_ratio)
    
    if pipeline_output:
        return negative_marker_purity
    else:
        # Calculate per gene and per cell type purities
        purities = (ratio_celltype_sc - ratio_celltype_sp).clip(0,None)
        purities.values[~neg_marker_mask] = np.nan
        purities = purities.loc[~(purities.isnull().all(axis=1)), ~(purities.isnull().all(axis=0))]    
        purity_per_gene = purities.mean(axis=0, skipna=True)
        purity_per_celltype = purities.mean(axis=1, skipna=True)
        
        return negative_marker_purity, purity_per_gene, purity_per_celltype


def negative_marker_purity_reads(adata_sp: AnnData, adata_sc: AnnData, key: str='celltype', pipeline_output: bool=True):
    """ Negative marker purity aims to measure read leakeage between cells in spatial datasets. 
    
    For this, we calculate the increase in reads assigned in spatial datasets to pairs of genes-celltyes with no/very low expression in scRNAseq
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    negative marker purity : float
       Increase in proportion of reads assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    """   
    
    # Set threshold parameters
    min_number_cells=10 #minimum number of cells belonging to a cluster to consider it in the analysis
    minimum_exp=0.005 #maximum relative expression allowed in a gene in a cluster to consider the gene-celltype pair the analysis 
    
    celltypes, adata_sp, adata_sc  = get_eligible_celltypes(adata_sp, adata_sc, key)

    # Return nan if too few cell types were found
    if len(celltypes) < 2:
        print("Not enough cell types (>1) eligible to calculate negative marker purity")
        negative_marker_purity = 'nan'
        if pipeline_output==True:
            return negative_marker_purity
        else:
            return negative_marker_purity, None, None
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp = adata_sp[adata_sp.obs[key].isin(celltypes)]
    
    # Get mean expression per cell type
    exp_sc = pd.DataFrame(adata_sc.layers['raw'],columns=adata_sp.var_names)
    exp_sp = pd.DataFrame(adata_sp.layers['raw'],columns=adata_sp.var_names)
    exp_sc['celltype'] = list(adata_sc.obs[key])
    exp_sp['celltype'] = list(adata_sp.obs[key])
    mean_celltype_sc = exp_sc.groupby('celltype').mean()
    mean_celltype_sp = exp_sp.groupby('celltype').mean()
    
    # Get mean expressions relative to mean over cell types (this will be used for filtering based on minimum_exp)
    mean_ct_sc_rel = mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)
    mean_ct_sp_rel = mean_celltype_sp.div(mean_celltype_sp.mean(axis=0),axis=1)
    # Potential TODO: this step is actually a tricky one 
    #       - at first it was normalised wrt sum instead of mean, imo it's not a good choice since for a high number of
    #         cell types we then always get many cell types < minimum_exp
    #       - we could normalize to the mean expression in cells that do express the gene probably best to have this 
    #         mean per cell type to account for cell type proportion differences. However, if spatial data has a lot of 
    #         leakage we add many more cells into the mean calculation which lowers the mean. 
    #       - Let's stick with the current one for now, however others should be explored at some point
    
    # Get normalized mean expressions over cell types (this will be summed up over negative cell types)
    mean_ct_sc_norm = mean_celltype_sc.div(mean_celltype_sc.sum(axis=0),axis=1)
    mean_ct_sp_norm = mean_celltype_sp.div(mean_celltype_sp.sum(axis=0),axis=1)
    # Explanation: The idea is to measure which ratio of mean expressions is shifted towards negative clusters.
    #              With this approach the metric stays between 0 and 1
    
    
    # Get gene-cell type pairs with negative marker expression
    neg_marker_mask = np.array(mean_ct_sc_rel < minimum_exp)
    
    # Return nan if no negative markers were found
    if np.sum(neg_marker_mask) < 1:
        print("No negative markers were found in the sc data reference.")
        negative_marker_purity = 'nan'
        if pipeline_output==True:
            return negative_marker_purity
        else:
            return negative_marker_purity, None, None
    
    # Get marker expressions in negative marker-cell type pairs
    lowvals_sc = np.full_like(neg_marker_mask, np.nan, dtype=np.float32)
    lowvals_sc = mean_ct_sc_norm.values[neg_marker_mask]
    lowvals_sp = mean_ct_sp_norm.values[neg_marker_mask]
    
    # Take the mean over the normalized expressions of the genes' negative cell types
    mean_sc_lowexp = np.nanmean(lowvals_sc)
    mean_sp_lowexp = np.nanmean(lowvals_sp)
    
    # Calculate summary metric
    negative_marker_purity = 1
    if mean_sp_lowexp > mean_sc_lowexp:
        negative_marker_purity -= (mean_sp_lowexp - mean_sc_lowexp)
    
    
    if pipeline_output:
        return negative_marker_purity
    else:
        # Calculate per gene and per cell type purities
        purities = (mean_ct_sp_norm - mean_ct_sc_norm).clip(0,None)
        purities.values[~neg_marker_mask] = np.nan
        purities = purities.loc[~(purities.isnull().all(axis=1)), ~(purities.isnull().all(axis=0))]
        purity_per_gene = purities.mean(axis=0, skipna=True)
        purity_per_celltype = purities.mean(axis=1, skipna=True)
        
        return negative_marker_purity, purity_per_gene, purity_per_celltype


def get_negative_marker_dict(adata_sp: AnnData, adata_sc: AnnData, key: str='celltype'):
    """Add dictionary of negative marker genes for different celltypes.
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs

    Returns
    -------
    neg_marker_dict : dict[string: list]
        Dictionary with celltypes as keys and corresponding negative marker genes as values
    """
    max_ratio_cells=0.005 # maximum ratio of cells expressing a marker to call it a negative marker gene-ct pair
           
    celltypes, adata_sp, adata_sc  = get_eligible_celltypes(adata_sp, adata_sc, key)  
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    genes = adata_sc.var_names
    
    # Get ratio of positive cells per cell type
    pos_exp_sc = pd.DataFrame(adata_sc.layers["raw"] > 0,columns=adata_sp.var_names)     
    pos_exp_sc['celltype'] = list(adata_sc.obs[key])
    ratio_celltype_sc = pos_exp_sc.groupby('celltype').mean()
    
    # Get gene-cell type pairs with negative marker expression
    neg_marker_mask = np.array(ratio_celltype_sc < max_ratio_cells)
    

    neg_marker_mask = pd.DataFrame(neg_marker_mask, columns = genes, index = ratio_celltype_sc.index)   
    neg_marker_dict = {celltype: list(row.index[row]) for celltype, row in neg_marker_mask.iterrows()}

    return neg_marker_dict


def get_spot_assignment_col(adata_sp: AnnData, adata_sc: AnnData):

    """Add column spot_assignment to adata_sp.uns["spots"].

       spot_assignment is "spot in wrong celltype", if celltype of cell is in the keys of the neg_marker_dict and gene is a negative marker for the celltype. 
       spot_assignment is "unassigned" if the corresponding cell entry is NaN.
       spot_assignment is "no negative marker" if gene is  not a negative marker for any celltype in neg_marker_dict.
       spot_assignment is "spot in correct celltype" otherwise.
      
       Parameters
       ----------
       adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
       adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    """ 
    #check that genes in spatial data is subset of genes in single cell data
    adata_sp = adata_sp[:,adata_sp.var_names.isin(adata_sc.var_names)]

    df = adata_sp.uns["spots"]  
   
    neg_marker_dict = get_negative_marker_dict(adata_sp, adata_sc)

    celltypes = list(neg_marker_dict.keys())   

    df["spot_assignment"] = "spot in correct celltype"

    all_neg_markers = set(itertools.chain(*neg_marker_dict.values()))
    #gene is not a negative marker for any celltype in neg_marker_dict and celltype is not in celltypes 
    df.loc[(~df["Gene"].isin(all_neg_markers))& (~df["celltype"].isin(celltypes)), "spot_assignment"] = "no negative marker"  

    for ct in celltypes:
      df.loc[(df["Gene"].isin(neg_marker_dict[ct])) & (df["celltype"] == ct), 'spot_assignment'] = "spot in wrong celltype" 

    df.loc[df["cell"]!=df["cell"],"spot_assignment"] = "unassigned"   #check for NaNs
